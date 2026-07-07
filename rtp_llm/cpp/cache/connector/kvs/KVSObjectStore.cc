#include "rtp_llm/cpp/cache/connector/kvs/KVSObjectStore.h"

#include <functional>
#include <sstream>
#include <utility>

namespace rtp_llm {
namespace {

class ScopeGuard {
public:
    explicit ScopeGuard(std::function<void()> fn): fn_(std::move(fn)) {}
    ~ScopeGuard() {
        if (active_) {
            try {
                fn_();
            } catch (...) {
            }
        }
    }

    void dismiss() {
        active_ = false;
    }

private:
    std::function<void()> fn_;
    bool                  active_{true};
};

}  // namespace

KVSObjectStore::KVSObjectStore(KVSObjectStoreConfig config, std::shared_ptr<KVSObjectBackend> backend):
    config_(std::move(config)), backend_(std::move(backend)) {}

std::string KVSObjectStore::makeKey(const KVSBlockIdentity& identity) const {
    std::ostringstream oss;
    oss << config_.object_namespace << "/" << config_.cache_key_version << "/" << identity.cache_key << "/g"
        << identity.group_id;
    return oss.str();
}

std::optional<KVSReadHandle>
KVSObjectStore::acquire(const std::vector<KVSObjectBuffer>& objects, const std::string& trace_id) {
    if (!backend_ || objects.empty()) {
        return std::nullopt;
    }
    return backend_->get(objectKeys(objects), trace_id);
}

bool KVSObjectStore::fetch(const KVSReadHandle&                handle,
                           const std::vector<KVSObjectBuffer>& objects,
                           const std::string&                  trace_id) {
    if (!backend_ || objects.empty()) {
        return false;
    }
    const auto keys = objectKeys(objects);
    return handle.containsAll(keys) && backend_->fetch(handle, keys, trace_id)
           && backend_->complete(handle, keys, trace_id);
}

bool KVSObjectStore::loadLocal(const std::vector<KVSObjectBuffer>& objects, const std::string& trace_id) {
    if (!backend_) {
        return false;
    }
    const auto keys = objectKeys(objects);
    auto handle = backend_->get(keys, trace_id);
    if (!handle.has_value()) {
        return false;
    }
    auto backend = backend_;
    ScopeGuard release_guard([backend, handle = *handle, &trace_id]() { backend->release(handle, trace_id); });
    if (!handle->containsAll(keys)) {
        return false;
    }
    return backend_->load(*handle, objects);
}

bool KVSObjectStore::write(const std::vector<KVSObjectBuffer>& objects, const std::string& trace_id) {
    if (!backend_) {
        return false;
    }
    const auto keys  = objectKeys(objects);
    const auto sizes = objectSizes(objects);
    auto handle = backend_->create(keys, sizes, trace_id);
    if (!handle.has_value()) {
        return false;
    }
    auto backend = backend_;
    ScopeGuard discard_guard([backend, handle = *handle, &trace_id]() { backend->discard(handle, trace_id); });
    if (!handle->containsAll(keys)) {
        return false;
    }
    if (!backend_->store(*handle, objects)) {
        return false;
    }
    if (!backend_->complete(*handle, keys, trace_id)) {
        return false;
    }
    discard_guard.dismiss();
    backend_->release(*handle, trace_id);
    return true;
}

void KVSObjectStore::release(const KVSReadHandle& handle, const std::string& trace_id) {
    if (backend_ && handle.valid()) {
        backend_->release(handle, trace_id);
    }
}

std::vector<std::string> KVSObjectStore::objectKeys(const std::vector<KVSObjectBuffer>& buffers) const {
    std::vector<std::string> keys;
    keys.reserve(buffers.size());
    for (const auto& buffer : buffers) {
        keys.push_back(buffer.object_key);
    }
    return keys;
}

std::vector<size_t> KVSObjectStore::objectSizes(const std::vector<KVSObjectBuffer>& buffers) const {
    std::vector<size_t> sizes;
    sizes.reserve(buffers.size());
    for (const auto& buffer : buffers) {
        sizes.push_back(buffer.totalBytes());
    }
    return sizes;
}

}  // namespace rtp_llm
