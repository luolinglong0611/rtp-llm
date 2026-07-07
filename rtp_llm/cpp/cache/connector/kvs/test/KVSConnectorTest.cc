#include "rtp_llm/cpp/cache/connector/kvs/KVSConnector.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/spec/MHAKVCacheSpec.h"

namespace rtp_llm {
namespace {

class FakeKVSObjectBackend: public KVSObjectBackend {
public:
    std::optional<KVSReadHandle> get(const std::vector<std::string>& object_keys,
                                      const std::string&              trace_id) override {
        ++get_count;
        get_trace = trace_id;
        KVSReadHandle handle;
        handle.handle_id = "read-lease";
        for (const auto& key : object_keys) {
            if (objects.count(key) != 0) {
                handle.object_keys.insert(key);
            }
        }
        return handle;
    }

    std::optional<KVSReadHandle> create(const std::vector<std::string>& object_keys,
                                         const std::vector<size_t>&      object_sizes,
                                         const std::string&              trace_id) override {
        create_trace  = trace_id;
        created_keys  = object_keys;
        created_sizes = object_sizes;
        KVSReadHandle handle;
        handle.handle_id = "create-lease";
        for (const auto& key : object_keys) {
            handle.object_keys.insert(key);
        }
        return handle;
    }

    bool fetch(const KVSReadHandle& handle,
               const std::vector<std::string>& object_keys,
               const std::string&              trace_id) override {
        ++fetch_count;
        fetch_trace = trace_id;
        fetched_keys.insert(fetched_keys.end(), object_keys.begin(), object_keys.end());
        for (const auto& key : object_keys) {
            if (fetch_fail_keys.count(key) != 0) {
                return false;
            }
        }
        return handle.containsAll(object_keys);
    }

    bool load(const KVSReadHandle& handle, const std::vector<KVSObjectBuffer>& dst_buffers) override {
        ++load_count;
        for (const auto& dst : dst_buffers) {
            if (!handle.contains(dst.object_key) || objects.count(dst.object_key) == 0) {
                return false;
            }
            size_t copied = 0;
            for (const auto& buffer : dst.buffers) {
                const size_t object_offset = dst.partial ? buffer.object_offset : copied;
                if (object_offset + buffer.size > objects.at(dst.object_key).size()) {
                    return false;
                }
                auto* dst_ptr = reinterpret_cast<char*>(buffer.addr);
                std::memcpy(dst_ptr, objects.at(dst.object_key).data() + object_offset, buffer.size);
                copied += buffer.size;
            }
        }
        return true;
    }

    bool store(const KVSReadHandle& handle, const std::vector<KVSObjectBuffer>& src_buffers) override {
        for (const auto& src : src_buffers) {
            if (!handle.contains(src.object_key)) {
                return false;
            }
            std::string data(src.totalBytes(), '\0');
            size_t      copied = 0;
            for (const auto& buffer : src.buffers) {
                const auto* src_ptr = reinterpret_cast<const char*>(buffer.addr);
                std::memcpy(data.data() + copied, src_ptr, buffer.size);
                copied += buffer.size;
            }
            objects[src.object_key] = std::move(data);
        }
        return true;
    }

    bool complete(const KVSReadHandle& handle,
                  const std::vector<std::string>& object_keys,
                  const std::string&              trace_id) override {
        ++complete_count;
        complete_trace = trace_id;
        return handle.containsAll(object_keys);
    }

    void release(const KVSReadHandle& handle, const std::string& trace_id) override {
        released.emplace_back(handle.handle_id, trace_id);
    }

    void discard(const KVSReadHandle& handle, const std::string& trace_id) override {
        discarded.emplace_back(handle.handle_id, trace_id);
    }

    int get_count{0};
    int fetch_count{0};
    int complete_count{0};
    int load_count{0};
    std::string get_trace;
    std::string create_trace;
    std::string fetch_trace;
    std::string complete_trace;
    std::vector<std::string> fetched_keys;
    std::vector<std::string> created_keys;
    std::vector<size_t> created_sizes;
    std::vector<std::pair<std::string, std::string>> released;
    std::vector<std::pair<std::string, std::string>> discarded;
    std::unordered_set<std::string> fetch_fail_keys;
    std::unordered_map<std::string, std::string> objects;
};

CacheConfig makeCacheConfig() {
    CacheConfig config;
    config.layer_num     = 1;
    config.layer_all_num = 1;
    auto spec            = std::make_shared<MHAKVCacheSpec>();
    spec->tag            = "kv";
    spec->layers         = {0};
    spec->dtype          = TYPE_FP16;

    GroupBase group;
    group.spec       = spec;
    group.layer_ids  = {0};
    group.policy.group_type = CacheGroupType::FULL;
    config.groups.push_back(group);
    config.layers.resize(1);
    config.tag_to_gid["kv"] = 0;
    return config;
}

CacheConfig makeTwoLayerCacheConfig() {
    CacheConfig config;
    config.layer_num     = 2;
    config.layer_all_num = 2;
    auto spec            = std::make_shared<MHAKVCacheSpec>();
    spec->tag            = "kv";
    spec->layers         = {0, 1};
    spec->dtype          = TYPE_FP16;

    GroupBase group;
    group.spec       = spec;
    group.layer_ids  = {0, 1};
    group.policy.group_type = CacheGroupType::FULL;
    config.groups.push_back(group);
    config.layers.resize(2);
    config.tag_to_gid["kv"] = 0;
    return config;
}

std::shared_ptr<KVCacheResource> makeResource(const std::vector<KVSCacheKey>& cache_keys,
                                              const std::vector<BlockIdxType>& block_ids,
                                              bool                            last_block_aligned = true) {
    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(1, 1, {{0}});
    resource->setCacheKeys(CacheKeysType(cache_keys.begin(), cache_keys.end()));
    resource->mutableBlockIds(0, 0).assign(block_ids);
    resource->setLastBlockAligned(last_block_aligned);
    return resource;
}

std::shared_ptr<KVCacheResource> makeTwoLayerResource(const std::vector<KVSCacheKey>& cache_keys,
                                                      const std::vector<BlockIdxType>& block_ids) {
    auto resource = std::make_shared<KVCacheResource>();
    resource->initGroups(1, 2, {{0}, {0}});
    resource->setCacheKeys(CacheKeysType(cache_keys.begin(), cache_keys.end()));
    resource->mutableBlockIds(0).assign(block_ids);
    resource->setLastBlockAligned(true);
    return resource;
}

std::string objectKey(KVSCacheKey cache_key) {
    return "rtp/v1/" + std::to_string(cache_key) + "/g0";
}

KVSConnectorConfig makeConnectorConfig() {
    KVSConnectorConfig config;
    config.object_namespace  = "rtp";
    config.cache_key_version = "v1";
    config.worker_thread_num = 1;
    config.worker_queue_size = 16;
    config.inline_execute    = true;
    return config;
}

std::shared_ptr<KVSObjectStore> makeObjectStore(const std::shared_ptr<KVSObjectBackend>& backend) {
    KVSObjectStoreConfig config;
    config.object_namespace  = "rtp";
    config.cache_key_version = "v1";
    return std::make_shared<KVSObjectStore>(std::move(config), backend);
}

KVSConnector::BlockBufferResolver
makeBlockBufferResolver(const std::shared_ptr<std::unordered_map<int, std::vector<char>>>& blocks) {
    return [blocks](int layer_id, int group_id, BlockIdxType block_id) {
        (void)layer_id;
        (void)group_id;
        auto iter = blocks->find(block_id);
        if (iter == blocks->end()) {
            return std::vector<BlockInfo>{};
        }
        BlockInfo info;
        info.addr       = iter->second.data();
        info.size_bytes = iter->second.size();
        return std::vector<BlockInfo>{info};
    };
}

KVSConnector::BlockBufferResolver
makeLayerBlockBufferResolver(const std::shared_ptr<std::unordered_map<int, std::vector<char>>>& layer_blocks) {
    return [layer_blocks](int layer_id, int group_id, BlockIdxType block_id) {
        (void)group_id;
        auto iter = layer_blocks->find(layer_id);
        if (iter == layer_blocks->end() || block_id <= 0) {
            return std::vector<BlockInfo>{};
        }
        BlockInfo info;
        info.addr       = iter->second.data();
        info.size_bytes = iter->second.size();
        return std::vector<BlockInfo>{info};
    };
}

TEST(KVSAsyncContextTest, SuccessCompletesWithMatchedBlockCount) {
    KVSAsyncContext context;
    EXPECT_FALSE(context.done());
    EXPECT_FALSE(context.success());

    context.markRunning();
    EXPECT_FALSE(context.done());

    context.markSuccess(3);
    EXPECT_TRUE(context.done());
    EXPECT_TRUE(context.success());
    EXPECT_EQ(context.matchedBlockCount(), 3);
    EXPECT_TRUE(context.errorInfo().ok());
}

TEST(KVSAsyncContextTest, FailureCompletesWithErrorInfo) {
    KVSAsyncContext context;
    context.markFailed("thread pool full");

    EXPECT_TRUE(context.done());
    EXPECT_FALSE(context.success());
    EXPECT_TRUE(context.errorInfo().hasError());
    EXPECT_EQ(context.errorInfo().ToString(), "thread pool full");
}

TEST(KVSAsyncContextTest, WaitDoneBlocksUntilCompletion) {
    KVSAsyncContext context;
    std::atomic<bool> wait_returned{false};

    std::thread waiter([&]() {
        context.waitDone();
        wait_returned = true;
    });

    context.markSuccess(1);
    waiter.join();
    EXPECT_TRUE(wait_returned.load());
}

TEST(KVSConnectorTest, AsyncWriteWritesCompleteBlocks) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    auto blocks       = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1]      = {'a', 'b', 'c'};
    (*blocks)[2]      = {'d', 'e', 'f'};

    KVSConnector connector(
        cache_config, makeConnectorConfig(), makeObjectStore(backend), makeBlockBufferResolver(blocks));
    ASSERT_TRUE(connector.init());
    auto context = connector.asyncWrite(makeResource({101, 102}, {1, 2}), nullptr);
    context->waitDone();

    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(backend->objects[objectKey(101)], "abc");
    EXPECT_EQ(backend->objects[objectKey(102)], "def");
    auto kvs_context = std::dynamic_pointer_cast<KVSAsyncContext>(context);
    ASSERT_TRUE(kvs_context);
    EXPECT_EQ(kvs_context->matchedBlockCount(), 2);
}

TEST(KVSConnectorTest, AsyncWriteAggregatesRequiredLayersIntoOneGroupObject) {
    auto cache_config = makeTwoLayerCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    auto layer_blocks = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*layer_blocks)[0] = {'a', 'b'};
    (*layer_blocks)[1] = {'c', 'd'};

    KVSConnector connector(
        cache_config, makeConnectorConfig(), makeObjectStore(backend), makeLayerBlockBufferResolver(layer_blocks));
    ASSERT_TRUE(connector.init());
    auto context = connector.asyncWrite(makeTwoLayerResource({101}, {1}), nullptr);
    context->waitDone();

    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(backend->created_keys, std::vector<std::string>{objectKey(101)});
    EXPECT_EQ(backend->created_sizes, std::vector<size_t>{4});
    EXPECT_EQ(backend->objects[objectKey(101)], "abcd");
}

TEST(KVSConnectorTest, AsyncMatchStopsAtIncompleteBlockPlan) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    auto blocks       = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1]      = {'a', 'b', 'c'};
    backend->objects[objectKey(101)] = "abc";
    backend->objects[objectKey(102)] = "def";

    KVSConnector connector(
        cache_config, makeConnectorConfig(), makeObjectStore(backend), makeBlockBufferResolver(blocks));
    ASSERT_TRUE(connector.init());
    auto context = connector.asyncMatch(makeResource({101, 102}, {1, NULL_BLOCK_IDX}), nullptr);
    context->waitDone();

    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(context->matchedBlockCount(), 1);
}

TEST(KVSConnectorTest, AsyncMatchGetsLeaseButDoesNotFetchOrResolveCoordinatorBuffers) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "abc";

    std::atomic<int> resolve_count{0};
    KVSConnector::BlockBufferResolver resolver = [&resolve_count](int layer_id, int group_id, BlockIdxType block_id) {
        (void)layer_id;
        (void)group_id;
        (void)block_id;
        ++resolve_count;
        return std::vector<BlockInfo>{};
    };

    KVSConnector connector(cache_config, makeConnectorConfig(), makeObjectStore(backend), std::move(resolver));
    ASSERT_TRUE(connector.init());
    auto context = connector.asyncMatch(makeResource({101}, {1}), nullptr);
    context->waitDone();

    EXPECT_TRUE(context->success()) << context->errorInfo().ToString();
    EXPECT_EQ(context->matchedBlockCount(), 1);
    EXPECT_EQ(resolve_count.load(), 0);
    EXPECT_EQ(backend->get_count, 1);
    EXPECT_EQ(backend->fetch_count, 0);
    EXPECT_EQ(backend->complete_count, 0);
    EXPECT_EQ(backend->load_count, 0);
    EXPECT_TRUE(backend->released.empty());

    context.reset();
    ASSERT_EQ(backend->released.size(), 1);
    EXPECT_EQ(backend->released[0].first, "read-lease");
}

TEST(KVSConnectorTest, AsyncReadFetchesBeforeSendingReadPlan) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "abc";

    auto blocks  = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1] = {0, 0, 0};

    std::vector<KVSConnector::KVSReadObjectPlan> captured_objects;
    KVSConnector::KVSReadPlanSender sender =
        [&captured_objects](const std::vector<KVSConnector::KVSReadObjectPlan>& objects,
                            const std::string&                                  trace_id) {
        (void)trace_id;
        captured_objects = objects;
        return true;
    };

    KVSConnector connector(cache_config,
                           makeConnectorConfig(),
                           makeObjectStore(backend),
                           makeBlockBufferResolver(blocks),
                           {},
                           std::move(sender));
    ASSERT_TRUE(connector.init());
    auto match_context = connector.asyncMatch(makeResource({101}, {1}), nullptr);
    match_context->waitDone();
    ASSERT_TRUE(match_context->success());

    auto read_context = connector.asyncRead(makeResource({101}, {1}), nullptr, match_context, 0, 1);
    read_context->waitDone();

    EXPECT_TRUE(read_context->success()) << read_context->errorInfo().ToString();
    auto kvs_read_context = std::dynamic_pointer_cast<KVSAsyncContext>(read_context);
    ASSERT_TRUE(kvs_read_context);
    EXPECT_EQ(kvs_read_context->matchedBlockCount(), 1);
    EXPECT_EQ(backend->fetch_count, 1);
    EXPECT_EQ(backend->complete_count, 1);
    ASSERT_EQ(backend->released.size(), 1);
    EXPECT_EQ(backend->released[0].first, "read-lease");
    ASSERT_EQ(captured_objects.size(), 1);
    EXPECT_EQ(captured_objects[0].object_key, objectKey(101));
    ASSERT_EQ(captured_objects[0].buffers.size(), 1);
    EXPECT_EQ(captured_objects[0].buffers[0].layer_id, 0);
    EXPECT_EQ(captured_objects[0].buffers[0].group_id, 0);
    EXPECT_EQ(captured_objects[0].buffers[0].block_id, 1);
}

TEST(KVSConnectorTest, AsyncReadTreatsFetchFailureAsPrefixMiss) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "abc";
    backend->objects[objectKey(102)] = "def";
    backend->fetch_fail_keys.insert(objectKey(102));

    auto blocks  = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1] = {'\0', '\0', '\0'};
    (*blocks)[2] = {'\0', '\0', '\0'};

    std::vector<KVSConnector::KVSReadObjectPlan> captured_objects;
    KVSConnector::KVSReadPlanSender sender =
        [&captured_objects](const std::vector<KVSConnector::KVSReadObjectPlan>& objects,
                            const std::string&                                  trace_id) {
        (void)trace_id;
        captured_objects = objects;
        return true;
    };

    KVSConnector connector(cache_config,
                           makeConnectorConfig(),
                           makeObjectStore(backend),
                           makeBlockBufferResolver(blocks),
                           {},
                           std::move(sender));
    ASSERT_TRUE(connector.init());
    auto resource      = makeResource({101, 102}, {1, 2});
    auto match_context = connector.asyncMatch(resource, nullptr);
    match_context->waitDone();
    ASSERT_TRUE(match_context->success());
    ASSERT_EQ(match_context->matchedBlockCount(), 2);

    auto read_context = connector.asyncRead(resource, nullptr, match_context, 0, 2);
    read_context->waitDone();

    EXPECT_TRUE(read_context->success()) << read_context->errorInfo().ToString();
    auto kvs_read_context = std::dynamic_pointer_cast<KVSAsyncContext>(read_context);
    ASSERT_TRUE(kvs_read_context);
    EXPECT_EQ(kvs_read_context->matchedBlockCount(), 1);
    ASSERT_EQ(captured_objects.size(), 1);
    EXPECT_EQ(captured_objects[0].object_key, objectKey(101));
    EXPECT_EQ(resource->remoteReuseBlockNum(), 1);
}

TEST(KVSConnectorTest, AsyncReadPlanCarriesObjectOffsetsForRequiredLayers) {
    auto cache_config = makeTwoLayerCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "abcd";

    auto layer_blocks = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*layer_blocks)[0] = {0, 0};
    (*layer_blocks)[1] = {0, 0};

    std::vector<KVSConnector::KVSReadObjectPlan> captured_objects;
    KVSConnector::KVSReadPlanSender sender =
        [&captured_objects](const std::vector<KVSConnector::KVSReadObjectPlan>& objects,
                            const std::string&                                  trace_id) {
        (void)trace_id;
        captured_objects = objects;
        return true;
    };

    KVSConnector connector(cache_config,
                           makeConnectorConfig(),
                           makeObjectStore(backend),
                           makeLayerBlockBufferResolver(layer_blocks),
                           {},
                           std::move(sender));
    ASSERT_TRUE(connector.init());
    auto resource      = makeTwoLayerResource({101}, {1});
    auto match_context = connector.asyncMatch(resource, nullptr);
    match_context->waitDone();
    ASSERT_TRUE(match_context->success());

    auto read_context = connector.asyncRead(resource, nullptr, match_context, 0, 1);
    read_context->waitDone();

    EXPECT_TRUE(read_context->success()) << read_context->errorInfo().ToString();
    ASSERT_EQ(captured_objects.size(), 1);
    ASSERT_EQ(captured_objects[0].buffers.size(), 2);
    EXPECT_EQ(captured_objects[0].buffers[0].object_offset, 0);
    EXPECT_EQ(captured_objects[0].buffers[1].object_offset, 2);
}

TEST(KVSConnectorTest, AsyncReadFromNonZeroBlockFetchesSuffixOnly) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "abc";
    backend->objects[objectKey(102)] = "def";
    backend->objects[objectKey(103)] = "ghi";

    auto blocks  = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1] = {0, 0, 0};
    (*blocks)[2] = {0, 0, 0};
    (*blocks)[3] = {0, 0, 0};

    std::vector<KVSConnector::KVSReadObjectPlan> captured_objects;
    KVSConnector::KVSReadPlanSender sender =
        [&captured_objects](const std::vector<KVSConnector::KVSReadObjectPlan>& objects,
                            const std::string&                                  trace_id) {
        (void)trace_id;
        captured_objects = objects;
        return true;
    };

    KVSConnector connector(cache_config,
                           makeConnectorConfig(),
                           makeObjectStore(backend),
                           makeBlockBufferResolver(blocks),
                           {},
                           std::move(sender));
    ASSERT_TRUE(connector.init());
    auto resource      = makeResource({101, 102, 103}, {1, 2, 3});
    auto match_context = connector.asyncMatch(resource, nullptr);
    match_context->waitDone();
    ASSERT_TRUE(match_context->success());
    ASSERT_EQ(match_context->matchedBlockCount(), 3);

    auto read_context = connector.asyncRead(resource, nullptr, match_context, 1, 2);
    read_context->waitDone();

    EXPECT_TRUE(read_context->success()) << read_context->errorInfo().ToString();
    auto kvs_read_context = std::dynamic_pointer_cast<KVSAsyncContext>(read_context);
    ASSERT_TRUE(kvs_read_context);
    EXPECT_EQ(kvs_read_context->matchedBlockCount(), 2);
    EXPECT_EQ(backend->fetched_keys, (std::vector<std::string>{objectKey(102), objectKey(103)}));
    ASSERT_EQ(captured_objects.size(), 2);
    EXPECT_EQ(captured_objects[0].object_key, objectKey(102));
    EXPECT_EQ(captured_objects[1].object_key, objectKey(103));
}

TEST(KVSConnectorTest, CopyCacheUsesExplicitObjectOffsets) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "xxabczz";
    auto blocks  = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1] = {0, 0, 0};

    KVSConnector connector(
        cache_config, makeConnectorConfig(), makeObjectStore(backend), makeBlockBufferResolver(blocks));
    ASSERT_TRUE(connector.init());

    KVSOperationRequestPB request;
    request.set_trace_id("trace");
    auto* item = request.add_read_items();
    item->set_object_key(objectKey(101));
    auto* buffer = item->add_buffers();
    buffer->set_layer_id(0);
    buffer->set_group_id(0);
    buffer->set_block_id(1);
    buffer->set_object_offset(2);

    KVSOperationResponsePB response;
    EXPECT_TRUE(connector.copyCache(request, response));
    EXPECT_TRUE(response.success());
    EXPECT_EQ((*blocks)[1], (std::vector<char>{'a', 'b', 'c'}));
}

TEST(KVSConnectorTest, CopyCacheLoadsLocalWithoutFetchOrComplete) {
    auto cache_config = makeCacheConfig();
    auto backend      = std::make_shared<FakeKVSObjectBackend>();
    backend->objects[objectKey(101)] = "abc";
    auto blocks  = std::make_shared<std::unordered_map<int, std::vector<char>>>();
    (*blocks)[1] = {'\0', '\0', '\0'};

    KVSConnector connector(
        cache_config, makeConnectorConfig(), makeObjectStore(backend), makeBlockBufferResolver(blocks));
    ASSERT_TRUE(connector.init());

    KVSOperationRequestPB request;
    request.set_trace_id("trace");
    auto* item = request.add_read_items();
    item->set_object_key(objectKey(101));
    auto* buffer = item->add_buffers();
    buffer->set_layer_id(0);
    buffer->set_group_id(0);
    buffer->set_block_id(1);

    KVSOperationResponsePB response;
    EXPECT_TRUE(connector.copyCache(request, response));
    EXPECT_TRUE(response.success());
    EXPECT_EQ((*blocks)[1], (std::vector<char>{'a', 'b', 'c'}));
    EXPECT_EQ(backend->fetch_count, 0);
    EXPECT_EQ(backend->complete_count, 0);
    EXPECT_EQ(backend->load_count, 1);
}

}  // namespace
}  // namespace rtp_llm
