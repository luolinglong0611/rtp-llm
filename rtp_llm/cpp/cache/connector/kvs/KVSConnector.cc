#include "rtp_llm/cpp/cache/connector/kvs/KVSConnector.h"

#include <algorithm>
#include <exception>
#include <functional>
#include <utility>

#include "autil/ThreadPool.h"
#include "rtp_llm/cpp/cache/connector/Meta.h"
#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

struct KVSConnectorTaskState {
    KVSConnectorTaskState(CacheConfig                       cache_config,
                          std::shared_ptr<KVSObjectStore>    store,
                          KVSConnector::BlockBufferResolver block_buffer_resolver):
        cache_config(std::move(cache_config)),
        store(std::move(store)),
        block_buffer_resolver(std::move(block_buffer_resolver)) {}

    CacheConfig                       cache_config;
    std::shared_ptr<KVSObjectStore>    store;
    KVSConnector::BlockBufferResolver block_buffer_resolver;
};

namespace {

const std::string kEmptyTraceId;

class HandleReleaseGuard {
public:
    explicit HandleReleaseGuard(std::shared_ptr<KVSMatchContext> context): context_(std::move(context)) {}

    ~HandleReleaseGuard() {
        context_->releaseHandle();
    }

private:
    std::shared_ptr<KVSMatchContext> context_;
};

struct BlockBufferPlan {
    std::vector<std::vector<KVSObjectBuffer>> block_buffers;
    bool                                      complete = true;
};

struct KVSBlockObjectPlan {
    std::vector<std::vector<KVSConnector::KVSReadObjectPlan>> block_objects;
    bool                                                      complete = true;
};

bool isUsableBlock(BlockIdxType block_id) {
    return block_id > 0 && !isNullBlockIdx(block_id);
}

const std::string& traceId(const std::shared_ptr<Meta>& meta) {
    if (!meta) {
        return kEmptyTraceId;
    }
    return meta->trace_id();
}

int writableBlockCount(const KVCacheResource& resource) {
    int block_num = static_cast<int>(resource.cacheKeys().size());
    if (!resource.lastBlockAligned() && block_num > 0) {
        --block_num;
    }
    return std::max(0, block_num);
}

KVSObjectBuffer buildGroupObject(const KVSConnectorTaskState& state,
                                 const KVCacheResource&       resource,
                                 int                          block_index,
                                 KVSCacheKey                  cache_key,
                                 int                          group_id);

KVSConnector::KVSReadObjectPlan buildGroupReadObject(const KVSConnectorTaskState& state,
                                                     const KVCacheResource&       resource,
                                                     int                          block_index,
                                                     KVSCacheKey                  cache_key,
                                                     int                          group_id,
                                                     bool                         include_object_offsets);

BlockBufferPlan
buildObjectBuffers(const KVSConnectorTaskState& state,
                   const KVCacheResource& resource,
                   int start_block_index,
                   int block_num) {
    BlockBufferPlan plan;
    if (start_block_index < 0 || block_num <= 0) {
        plan.complete = false;
        return plan;
    }
    const auto& cache_keys = resource.cacheKeys();
    const int   end        = std::min<int>(start_block_index + block_num, static_cast<int>(cache_keys.size()));
    plan.block_buffers.reserve(static_cast<size_t>(std::max(0, end - start_block_index)));
    for (int block_pos = start_block_index; block_pos < end; ++block_pos) {
        std::vector<KVSObjectBuffer> one_block_buffers;
        bool                         one_block_complete = true;
        for (int group_id = 0; group_id < state.cache_config.groupNums(); ++group_id) {
            auto object_buffer =
                buildGroupObject(state, resource, block_pos, cache_keys[static_cast<size_t>(block_pos)], group_id);
            if (object_buffer.buffers.empty()) {
                one_block_complete = false;
                break;
            }
            one_block_buffers.push_back(std::move(object_buffer));
        }
        if (!one_block_complete || one_block_buffers.empty()) {
            plan.complete = false;
            break;
        }
        plan.block_buffers.push_back(std::move(one_block_buffers));
    }
    if (static_cast<int>(plan.block_buffers.size()) != std::max(0, end - start_block_index)) {
        plan.complete = false;
    }
    return plan;
}

KVSBlockObjectPlan buildBlockObjects(const KVSConnectorTaskState& state,
                                     const KVCacheResource&       resource,
                                     int                          start_block_index,
                                     int                          block_num,
                                     bool                         include_object_offsets = false) {
    KVSBlockObjectPlan plan;
    if (start_block_index < 0 || block_num <= 0) {
        plan.complete = false;
        return plan;
    }
    const auto& cache_keys = resource.cacheKeys();
    const int   end        = std::min<int>(start_block_index + block_num, static_cast<int>(cache_keys.size()));
    plan.block_objects.reserve(static_cast<size_t>(std::max(0, end - start_block_index)));
    for (int block_pos = start_block_index; block_pos < end; ++block_pos) {
        std::vector<KVSConnector::KVSReadObjectPlan> one_block_objects;
        bool                                         one_block_complete = true;
        for (int group_id = 0; group_id < state.cache_config.groupNums(); ++group_id) {
            auto object = buildGroupReadObject(state,
                                               resource,
                                               block_pos,
                                               cache_keys[static_cast<size_t>(block_pos)],
                                               group_id,
                                               include_object_offsets);
            if (object.buffers.empty()) {
                one_block_complete = false;
                break;
            }
            one_block_objects.push_back(std::move(object));
        }
        if (!one_block_complete || one_block_objects.empty()) {
            plan.complete = false;
            break;
        }
        plan.block_objects.push_back(std::move(one_block_objects));
    }
    if (static_cast<int>(plan.block_objects.size()) != std::max(0, end - start_block_index)) {
        plan.complete = false;
    }
    return plan;
}

std::vector<KVSObjectBuffer> flattenBlockObjects(const std::vector<std::vector<KVSObjectBuffer>>& buffers) {
    std::vector<KVSObjectBuffer> out;
    size_t                       total = 0;
    for (const auto& one_block : buffers) {
        total += one_block.size();
    }
    out.reserve(total);
    for (const auto& one_block : buffers) {
        out.insert(out.end(), one_block.begin(), one_block.end());
    }
    return out;
}

std::vector<std::vector<KVSObjectBuffer>> sliceBlockObjects(const std::vector<std::vector<KVSObjectBuffer>>& objects,
                                                            int start_block_index,
                                                            int block_num) {
    std::vector<std::vector<KVSObjectBuffer>> out;
    if (start_block_index < 0 || block_num <= 0 || start_block_index >= static_cast<int>(objects.size())) {
        return out;
    }
    const int end = std::min<int>(start_block_index + block_num, static_cast<int>(objects.size()));
    out.reserve(static_cast<size_t>(end - start_block_index));
    for (int i = start_block_index; i < end; ++i) {
        out.push_back(objects[static_cast<size_t>(i)]);
    }
    return out;
}

std::vector<KVSConnector::KVSReadObjectPlan>
flattenTpReadPlan(const std::vector<std::vector<KVSConnector::KVSReadObjectPlan>>& objects) {
    std::vector<KVSConnector::KVSReadObjectPlan> out;
    size_t                                       total = 0;
    for (const auto& one_block : objects) {
        total += one_block.size();
    }
    out.reserve(total);
    for (const auto& one_block : objects) {
        out.insert(out.end(), one_block.begin(), one_block.end());
    }
    return out;
}

std::vector<std::vector<KVSObjectBuffer>>
readPlanToBlockObjects(const std::vector<std::vector<KVSConnector::KVSReadObjectPlan>>& objects) {
    std::vector<std::vector<KVSObjectBuffer>> out;
    out.reserve(objects.size());
    for (const auto& one_block : objects) {
        std::vector<KVSObjectBuffer> one_block_buffers;
        one_block_buffers.reserve(one_block.size());
        for (const auto& object : one_block) {
            KVSObjectBuffer buffer;
            buffer.object_key = object.object_key;
            one_block_buffers.push_back(std::move(buffer));
        }
        out.push_back(std::move(one_block_buffers));
    }
    return out;
}

bool containsBlockObjects(const KVSReadHandle& handle, const std::vector<KVSObjectBuffer>& objects) {
    if (objects.empty()) {
        return false;
    }
    for (const auto& object : objects) {
        if (object.object_key.empty() || !handle.contains(object.object_key)) {
            return false;
        }
    }
    return true;
}

size_t countPrefixBlocks(const std::vector<std::vector<KVSObjectBuffer>>& block_objects,
                         const KVSReadHandle&                            handle) {
    size_t matched_blocks = 0;
    for (const auto& one_block_objects : block_objects) {
        if (!containsBlockObjects(handle, one_block_objects)) {
            break;
        }
        ++matched_blocks;
    }
    return matched_blocks;
}

size_t fetchReadyBlocks(KVSObjectStore&                                  store,
                        const KVSReadHandle&                             handle,
                        const std::vector<std::vector<KVSObjectBuffer>>& block_objects,
                        const std::string&                               trace_id) {
    size_t ready_blocks = 0;
    for (const auto& one_block_objects : block_objects) {
        if (!containsBlockObjects(handle, one_block_objects)) {
            break;
        }
        if (!store.fetch(handle, one_block_objects, trace_id)) {
            break;
        }
        ++ready_blocks;
    }
    return ready_blocks;
}

KVSObjectBuffer buildGroupObject(const KVSConnectorTaskState& state,
                                 const KVCacheResource&       resource,
                                 int                          block_index,
                                 KVSCacheKey                  cache_key,
                                 int                          group_id) {
    KVSObjectBuffer object_buffer;
    KVSBlockIdentity identity;
    identity.cache_key = cache_key;
    identity.group_id  = group_id;
    object_buffer.object_key = state.store->makeKey(identity);

    const auto& layer_ids = state.cache_config.layerIdsForGroup(static_cast<size_t>(group_id));
    for (int layer_id : layer_ids) {
        const auto& blocks = resource.blocks(layer_id, group_id);
        if (block_index >= static_cast<int>(blocks.size()) || !isUsableBlock(blocks[block_index])) {
            object_buffer.buffers.clear();
            return object_buffer;
        }

        const auto block_buffers = state.block_buffer_resolver(layer_id, group_id, blocks[block_index]);
        if (block_buffers.empty()) {
            object_buffer.buffers.clear();
            return object_buffer;
        }

        const auto buffer_count_before_layer = object_buffer.buffers.size();
        object_buffer.buffers.reserve(object_buffer.buffers.size() + block_buffers.size());
        for (const auto& block_buffer : block_buffers) {
            if (block_buffer.addr != nullptr && block_buffer.size_bytes > 0) {
                KVSBuffer buffer;
                buffer.addr    = reinterpret_cast<uint64_t>(block_buffer.addr);
                buffer.size    = block_buffer.size_bytes;
                buffer.is_cuda = block_buffer.is_cuda;
                object_buffer.buffers.push_back(buffer);
            }
        }
        if (object_buffer.buffers.size() == buffer_count_before_layer) {
            object_buffer.buffers.clear();
            return object_buffer;
        }
    }
    return object_buffer;
}

KVSConnector::KVSReadObjectPlan buildGroupReadObject(const KVSConnectorTaskState& state,
                                                     const KVCacheResource&       resource,
                                                     int                          block_index,
                                                     KVSCacheKey                  cache_key,
                                                     int                          group_id,
                                                     bool                         include_object_offsets) {
    KVSConnector::KVSReadObjectPlan object;
    KVSBlockIdentity identity;
    identity.cache_key = cache_key;
    identity.group_id  = group_id;
    object.object_key  = state.store->makeKey(identity);

    size_t      object_offset = 0;
    const auto& layer_ids     = state.cache_config.layerIdsForGroup(static_cast<size_t>(group_id));
    for (int layer_id : layer_ids) {
        const auto& blocks = resource.blocks(layer_id, group_id);
        if (block_index >= static_cast<int>(blocks.size()) || !isUsableBlock(blocks[block_index])) {
            object.buffers.clear();
            return object;
        }
        const auto block_id = blocks[block_index];
        object.buffers.push_back(KVSConnector::KVSReadBufferSpec{layer_id, group_id, block_id, object_offset});

        if (!include_object_offsets) {
            continue;
        }
        const auto block_buffers = state.block_buffer_resolver(layer_id, group_id, block_id);
        size_t     layer_bytes   = 0;
        for (const auto& block_buffer : block_buffers) {
            if (block_buffer.addr != nullptr && block_buffer.size_bytes > 0) {
                layer_bytes += block_buffer.size_bytes;
            }
        }
        if (layer_bytes == 0) {
            object.buffers.clear();
            return object;
        }
        object_offset += layer_bytes;
    }
    return object;
}

KVSObjectBuffer resolveTpReadPlan(const KVSConnectorTaskState&           state,
                                  const KVSConnector::KVSReadObjectPlan& read_object) {
    KVSObjectBuffer object_buffer;
    object_buffer.object_key = read_object.object_key;
    object_buffer.partial    = true;
    for (const auto& spec : read_object.buffers) {
        size_t     object_offset = spec.object_offset;
        const auto block_buffers = state.block_buffer_resolver(spec.layer_id, spec.group_id, spec.block_id);
        if (block_buffers.empty()) {
            object_buffer.buffers.clear();
            return object_buffer;
        }
        const auto buffer_count_before_layer = object_buffer.buffers.size();
        object_buffer.buffers.reserve(object_buffer.buffers.size() + block_buffers.size());
        for (const auto& block_buffer : block_buffers) {
            if (block_buffer.addr != nullptr && block_buffer.size_bytes > 0) {
                KVSBuffer buffer;
                buffer.addr          = reinterpret_cast<uint64_t>(block_buffer.addr);
                buffer.size          = block_buffer.size_bytes;
                buffer.object_offset = object_offset;
                buffer.is_cuda       = block_buffer.is_cuda;
                object_buffer.buffers.push_back(buffer);
                object_offset += block_buffer.size_bytes;
            }
        }
        if (object_buffer.buffers.size() == buffer_count_before_layer) {
            object_buffer.buffers.clear();
            return object_buffer;
        }
    }
    return object_buffer;
}

void doMatch(const std::shared_ptr<const KVSConnectorTaskState>& state,
             const std::shared_ptr<KVCacheResource>&            resource,
             const std::shared_ptr<Meta>&                       meta,
             const std::shared_ptr<KVSMatchContext>&            context) {
    const int block_num = writableBlockCount(*resource);
    if (block_num <= 0) {
        context->markSuccess(0);
        return;
    }
    auto plan = buildBlockObjects(*state, *resource, 0, block_num);
    if (!plan.complete && plan.block_objects.empty()) {
        context->markSuccess(0);
        return;
    }
    auto block_objects = readPlanToBlockObjects(plan.block_objects);
    auto objects       = flattenBlockObjects(block_objects);
    auto handle        = state->store->acquire(objects, traceId(meta));
    if (!handle.has_value()) {
        context->markSuccess(0);
        return;
    }
    const auto matched = countPrefixBlocks(block_objects, *handle);
    context->setMatchResult(state->store, std::move(*handle), std::move(block_objects), traceId(meta), matched);
}

void doRead(const std::shared_ptr<const KVSConnectorTaskState>& state,
            const KVSConnector::KVSReadPlanSender&             read_plan_sender,
            const std::shared_ptr<KVCacheResource>&            resource,
            const std::shared_ptr<Meta>&                       meta,
            const std::shared_ptr<AsyncMatchContext>&          match_context,
            int                                                start_read_block_index,
            int                                                read_block_num,
            const std::shared_ptr<KVSAsyncContext>&            context) {
    if (!match_context->success()) {
        context->markFailed("KVS async read failed: match context failed");
        return;
    }
    auto kvs_match_context = std::dynamic_pointer_cast<KVSMatchContext>(match_context);
    if (!kvs_match_context || !kvs_match_context->hasHandle()) {
        context->markFailed("KVS async read failed: invalid KVS match context");
        return;
    }
    HandleReleaseGuard handle_guard(kvs_match_context);
    if (read_block_num <= 0) {
        context->markSuccess(0);
        return;
    }
    const auto matched_blocks = static_cast<int>(match_context->matchedBlockCount());
    if (start_read_block_index < 0 || start_read_block_index >= matched_blocks) {
        context->markSuccess(0);
        return;
    }
    const int readable_blocks = std::min(read_block_num, matched_blocks - start_read_block_index);
    auto fetch_objects = sliceBlockObjects(kvs_match_context->blockObjects(),
                                            start_read_block_index,
                                            readable_blocks);
    const int ready_blocks  = static_cast<int>(fetchReadyBlocks(
        *state->store, kvs_match_context->handle(), fetch_objects, traceId(meta)));
    if (ready_blocks <= 0) {
        context->markSuccess(0);
        return;
    }

    if (read_plan_sender) {
        auto plan = buildBlockObjects(*state, *resource, start_read_block_index, ready_blocks, true);
        if (!plan.complete || static_cast<int>(plan.block_objects.size()) != ready_blocks) {
            context->markFailed("KVS async read failed: incomplete read plan");
            return;
        }
        const auto objects = flattenTpReadPlan(plan.block_objects);
        if (!read_plan_sender(objects, traceId(meta))) {
            context->markFailed("KVS async read failed: read plan sender failed");
            return;
        }
    } else {
        auto plan = buildObjectBuffers(*state, *resource, start_read_block_index, ready_blocks);
        if (!plan.complete || static_cast<int>(plan.block_buffers.size()) != ready_blocks) {
            context->markFailed("KVS async read failed: incomplete block buffer plan");
            return;
        }
        const auto buffers = flattenBlockObjects(plan.block_buffers);
        if (!state->store->loadLocal(buffers, traceId(meta))) {
            context->markFailed("KVS async read failed: object store local load failed");
            return;
        }
    }
    resource->setRemoteReuseBlockNum(resource->remoteReuseBlockNum() + static_cast<size_t>(ready_blocks));
    context->markSuccess(ready_blocks);
}

void doWrite(const std::shared_ptr<const KVSConnectorTaskState>& state,
             const std::shared_ptr<KVCacheResource>&            resource,
             const std::shared_ptr<Meta>&                       meta,
             const std::shared_ptr<KVSAsyncContext>&            context) {
    const int block_num = writableBlockCount(*resource);
    if (block_num <= 0) {
        context->markSuccess(0);
        return;
    }
    auto plan = buildObjectBuffers(*state, *resource, 0, block_num);
    if (!plan.complete || static_cast<int>(plan.block_buffers.size()) != block_num) {
        context->markFailed("KVS async write failed: incomplete block buffer plan");
        return;
    }
    const auto buffers = flattenBlockObjects(plan.block_buffers);
    if (!state->store->write(buffers, traceId(meta))) {
        context->markFailed("KVS async write failed: object store write failed");
        return;
    }
    context->markSuccess(block_num);
}

}  // namespace

KVSAsyncContext::KVSAsyncContext(): error_info_(ErrorInfo::OkStatus()) {}

void KVSAsyncContext::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return state_ == State::SUCCESS || state_ == State::FAILED; });
}

bool KVSAsyncContext::done() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_ == State::SUCCESS || state_ == State::FAILED;
}

bool KVSAsyncContext::success() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_ == State::SUCCESS;
}

size_t KVSAsyncContext::matchedBlockCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return matched_block_count_;
}

ErrorInfo KVSAsyncContext::errorInfo() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_info_;
}

void KVSAsyncContext::markRunning() {
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = State::RUNNING;
}

void KVSAsyncContext::markSuccess(size_t matched_block_count) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        state_               = State::SUCCESS;
        matched_block_count_ = matched_block_count;
        error_info_          = ErrorInfo::OkStatus();
    }
    cv_.notify_all();
}

void KVSAsyncContext::markFailed(const std::string& message) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        state_      = State::FAILED;
        error_info_ = ErrorInfo(ErrorCode::UNKNOWN_ERROR, message);
    }
    cv_.notify_all();
}

KVSConnector::KVSConnector(const CacheConfig&                  cache_config,
                           KVSConnectorConfig                 config,
                           std::shared_ptr<KVSObjectStore>    store,
                           BlockBufferResolver                block_buffer_resolver,
                           std::vector<std::string>           tp_addrs,
                           KVSReadPlanSender                  read_plan_sender):
    cache_config_(cache_config),
    config_(std::move(config)),
    store_(std::move(store)),
    tp_addrs_(std::move(tp_addrs)),
    read_plan_sender_(std::move(read_plan_sender)) {
    task_state_ = std::make_shared<KVSConnectorTaskState>(cache_config_, store_, std::move(block_buffer_resolver));
}

KVSConnector::~KVSConnector() {
    if (worker_thread_pool_) {
        worker_thread_pool_->stop();
        worker_thread_pool_.reset();
    }
}

bool KVSConnector::init() {
    if (!task_state_->block_buffer_resolver) {
        RTP_LLM_LOG_WARNING("KVSConnector init failed, block buffer resolver is null");
        return false;
    }
    if (!store_) {
        RTP_LLM_LOG_WARNING("KVSConnector init failed, object store is null");
        return false;
    }
    if (cache_config_.groupNums() <= 0 || cache_config_.layer_all_num == 0) {
        RTP_LLM_LOG_WARNING("KVSConnector init failed, invalid cache config");
        return false;
    }
    if (!read_plan_sender_ && !tp_addrs_.empty()) {
        broadcast_manager_ = std::make_shared<BroadcastManager>(tp_addrs_);
        if (!broadcast_manager_->init()) {
            RTP_LLM_LOG_WARNING("KVSConnector init failed, broadcast manager init failed");
            broadcast_manager_.reset();
            return false;
        }
        read_plan_sender_ = [this](const std::vector<KVSReadObjectPlan>& objects, const std::string& trace_id) {
            return sendTpReadPlan(objects, trace_id);
        };
    }
    if (config_.inline_execute) {
        return true;
    }
    if (config_.worker_thread_num <= 0 || config_.worker_queue_size <= 0) {
        RTP_LLM_LOG_WARNING("KVSConnector init failed, invalid worker config, thread_num: %d, queue_size: %d",
                            config_.worker_thread_num,
                            config_.worker_queue_size);
        return false;
    }
    worker_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(
        config_.worker_thread_num, config_.worker_queue_size, nullptr, "KVSConnectorWorker");
    if (!worker_thread_pool_->start()) {
        RTP_LLM_LOG_WARNING("KVSConnector init failed, worker thread pool start failed");
        worker_thread_pool_.reset();
        return false;
    }
    return true;
}

KVSMatchContext::~KVSMatchContext() {
    releaseHandle();
}

void KVSMatchContext::setMatchResult(std::shared_ptr<KVSObjectStore>           store,
                                     KVSReadHandle                            handle,
                                     std::vector<std::vector<KVSObjectBuffer>> block_objects,
                                     std::string                              trace_id,
                                     size_t                                   matched_block_count) {
    store_         = std::move(store);
    handle_        = std::move(handle);
    block_objects_ = std::move(block_objects);
    trace_id_      = std::move(trace_id);
    has_handle_    = true;
    released_      = false;
    markSuccess(matched_block_count);
}

bool KVSMatchContext::hasHandle() const {
    return has_handle_ && !released_;
}

const KVSReadHandle& KVSMatchContext::handle() const {
    return handle_;
}

const std::vector<std::vector<KVSObjectBuffer>>& KVSMatchContext::blockObjects() const {
    return block_objects_;
}

void KVSMatchContext::releaseHandle() {
    if (!has_handle_ || released_) {
        return;
    }
    if (store_) {
        store_->release(handle_, trace_id_);
    }
    released_ = true;
}

bool KVSConnector::copyCache(const KVSOperationRequestPB& request, KVSOperationResponsePB& response) {
    std::vector<KVSObjectBuffer> buffers;
    buffers.reserve(static_cast<size_t>(request.read_items_size()));
    for (const auto& item : request.read_items()) {
        KVSReadObjectPlan object;
        object.object_key = item.object_key();
        object.buffers.reserve(static_cast<size_t>(item.buffers_size()));
        for (const auto& buffer : item.buffers()) {
            object.buffers.push_back(KVSReadBufferSpec{buffer.layer_id(),
                                                        buffer.group_id(),
                                                        buffer.block_id(),
                                                        static_cast<size_t>(buffer.object_offset())});
        }
        auto resolved = resolveTpReadPlan(*task_state_, object);
        if (resolved.buffers.empty()) {
            response.set_success(false);
            response.set_error_message("resolve KVS read buffers failed");
            return false;
        }
        buffers.push_back(std::move(resolved));
    }
    if (buffers.empty()) {
        response.set_success(false);
        response.set_error_message("KVS read request is empty");
        return false;
    }
    const bool success = store_->loadLocal(buffers, request.trace_id());
    response.set_success(success);
    if (!success) {
        response.set_error_message("KVS object store read failed");
    }
    return success;
}

std::shared_ptr<AsyncMatchContext> KVSConnector::asyncMatch(const std::shared_ptr<KVCacheResource>& resource,
                                                            const std::shared_ptr<Meta>&            meta) {
    auto context = std::make_shared<KVSMatchContext>();
    if (!resource || !store_) {
        context->markFailed("KVS async match failed: resource or store is null");
        return context;
    }
    auto state = task_state_;
    submitTask(context, [state, resource, meta, context]() { doMatch(state, resource, meta, context); });
    return context;
}

std::shared_ptr<AsyncContext> KVSConnector::asyncRead(const std::shared_ptr<KVCacheResource>&   resource,
                                                      const std::shared_ptr<Meta>&              meta,
                                                      const std::shared_ptr<AsyncMatchContext>& match_context,
                                                      int                                       start_read_block_index,
                                                      int                                       read_block_num) {
    auto context = std::make_shared<KVSAsyncContext>();
    if (!resource || !store_ || !match_context) {
        context->markFailed("KVS async read failed: resource, store or match context is null");
        return context;
    }
    auto state            = task_state_;
    auto read_plan_sender = read_plan_sender_;
    submitTask(context, [state,
                         read_plan_sender,
                         resource,
                         meta,
                         match_context,
                         start_read_block_index,
                         read_block_num,
                         context]() {
        doRead(state, read_plan_sender, resource, meta, match_context, start_read_block_index, read_block_num, context);
    });
    return context;
}

std::shared_ptr<AsyncContext> KVSConnector::asyncWrite(const std::shared_ptr<KVCacheResource>& resource,
                                                       const std::shared_ptr<Meta>&            meta) {
    auto context = std::make_shared<KVSAsyncContext>();
    if (!resource || !store_) {
        context->markFailed("KVS async write failed: resource or store is null");
        return context;
    }
    auto state = task_state_;
    submitTask(context, [state, resource, meta, context]() { doWrite(state, resource, meta, context); });
    return context;
}

std::shared_ptr<AsyncContext>
KVSConnector::asyncWriteByLayer(int layer_id, const std::shared_ptr<KVCacheConnectorLayerContext>& layer_context) {
    (void)layer_context;
    auto context = std::make_shared<KVSAsyncContext>();
    context->markFailed("KVSConnector asyncWriteByLayer is not implemented, layer_id: " + std::to_string(layer_id));
    return context;
}

bool KVSConnector::sendTpReadPlan(const std::vector<KVSReadObjectPlan>& objects, const std::string& trace_id) const {
    if (!broadcast_manager_) {
        RTP_LLM_LOG_WARNING("send KVS read plan failed, broadcast manager is null");
        return false;
    }

    KVSOperationRequestPB kvs_req;
    kvs_req.set_trace_id(trace_id);
    for (const auto& object : objects) {
        auto* item = kvs_req.add_read_items();
        item->set_object_key(object.object_key);
        for (const auto& buffer : object.buffers) {
            auto* spec = item->add_buffers();
            spec->set_layer_id(buffer.layer_id);
            spec->set_group_id(buffer.group_id);
            spec->set_block_id(buffer.block_id);
            spec->set_object_offset(buffer.object_offset);
        }
    }

    std::vector<FunctionRequestPB> requests;
    requests.reserve(broadcast_manager_->workerNum());
    for (size_t i = 0; i < broadcast_manager_->workerNum(); ++i) {
        FunctionRequestPB req;
        req.mutable_kvs_request()->CopyFrom(kvs_req);
        requests.emplace_back(std::move(req));
    }
    auto rpc_call = [](const std::shared_ptr<RpcService::Stub>&    stub,
                       const std::shared_ptr<grpc::ClientContext>& context,
                       const FunctionRequestPB&                    request,
                       grpc::CompletionQueue*                      completion_queue) {
        return stub->AsyncExecuteFunction(context.get(), request, completion_queue);
    };
    auto result =
        broadcast_manager_->broadcast<FunctionRequestPB, FunctionResponsePB>(requests, config_.timeout_ms, rpc_call);
    if (!result || !result->waitDone(config_.timeout_ms) || !result->success()) {
        RTP_LLM_LOG_WARNING("send KVS read plan failed, broadcast rpc failed");
        return false;
    }
    for (const auto& response : result->responses()) {
        if (!response.has_kvs_response() || !response.kvs_response().success()) {
            RTP_LLM_LOG_WARNING("send KVS read plan failed, worker response: %s", response.DebugString().c_str());
            return false;
        }
    }
    return true;
}

bool KVSConnector::submitTask(const std::shared_ptr<KVSAsyncContext>& context, std::function<void()> task) const {
    if (config_.inline_execute) {
        context->markRunning();
        try {
            task();
        } catch (const std::exception& e) {
            context->markFailed(std::string("KVSConnector task exception: ") + e.what());
        } catch (...) {
            context->markFailed("KVSConnector task unknown exception");
        }
        return true;
    }
    if (!worker_thread_pool_) {
        context->markFailed("KVSConnector worker thread pool is not initialized");
        return false;
    }
    context->markRunning();
    auto code = worker_thread_pool_->pushTask([context, task = std::move(task)]() mutable {
        try {
            task();
        } catch (const std::exception& e) {
            context->markFailed(std::string("KVSConnector task exception: ") + e.what());
        } catch (...) {
            context->markFailed("KVSConnector task unknown exception");
        }
    });
    if (code != autil::ThreadPoolBase::ERROR_NONE) {
        context->markFailed("KVSConnector worker thread pool push task failed, code: "
                            + std::to_string(static_cast<int>(code)));
        return false;
    }
    return true;
}

}  // namespace rtp_llm
