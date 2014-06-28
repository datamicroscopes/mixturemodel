#include <microscopes/mixture/model.hpp>
#include <microscopes/common/macros.hpp>
#include <microscopes/common/util.hpp>

#include <cassert>
#include <iostream>

using namespace std;
using namespace microscopes::common;
using namespace microscopes::models;
using namespace microscopes::mixture;

size_t
state::create_group(rng_t &rng)
{
  vector<shared_ptr<feature_group>> gdata;
  gdata.reserve(models_.size());
  for (auto &m : models_)
    gdata.emplace_back(m->create_feature_group(rng));
  const size_t gid = gcount_++;
  groups_[gid] = move(make_pair(0, move(gdata)));
  assert(!gempty_.count(gid));
  gempty_.insert(gid);
  return gid;
}

void
state::delete_group(size_t gid)
{
  auto it = groups_.find(gid);
  assert(it != groups_.end());
  assert(!it->second.first);
  assert(gempty_.count(gid));
  groups_.erase(it);
  gempty_.erase(gid);
  gremoved_++;
}

void
state::ensure_k_empty_groups(size_t k, rng_t &rng)
{
  // XXX: should allow for resampling
  if (empty_groups().size() == k)
    return;
  // XXX: NOT EFFICIENT
  vector<size_t> egids(gempty_.begin(), gempty_.end());
  for (auto egid : egids)
    delete_group(egid);
  for (size_t i = 0; i < k; i++)
    create_group(rng);
  assert( empty_groups().size() == k );
}

vector<runtime_type_info>
state::get_runtime_type_info() const
{
  vector<runtime_type_info> ret;
  ret.reserve(models_.size());
  for (const auto &m : models_)
    ret.push_back(m->get_runtime_type_info());
  return ret;
}

void
state::add_value(size_t gid, const dataview &view, rng_t &rng)
{
  assert(view.size() == assignments_.size());
  row_accessor acc = view.get();
  const size_t eid = view.index();
  add_value(gid, eid, acc, rng);
}

void
state::add_value(size_t gid, size_t eid, common::row_accessor &acc, common::rng_t &rng)
{
  assert(assignments_.at(eid) == -1);
  auto it = groups_.find(gid);
  assert(it != groups_.end());
  if (!it->second.first++) {
    assert(gempty_.count(gid));
    gempty_.erase(gid);
  } else {
    assert(!gempty_.count(gid));
  }
  acc.reset();
  assert(acc.nfeatures() == it->second.second.size());
  for (size_t i = 0; i < acc.nfeatures(); i++, acc.bump()) {
    if (unlikely(acc.ismasked()))
      continue;
    it->second.second[i]->add_value(*models_[i], acc, rng);
  }
  assignments_[eid] = gid;
}

size_t
state::remove_value(const common::dataview &view, rng_t &rng)
{
  assert(view.size() == assignments_.size());
  row_accessor acc = view.get();
  const size_t eid = view.index();
  return remove_value(eid, acc, rng);
}

size_t
state::remove_value(size_t eid, common::row_accessor &acc, common::rng_t &rng)
{
  assert(assignments_.at(eid) != -1);
  const size_t gid = assignments_[eid];
  auto it = groups_.find(gid);
  assert(it != groups_.end());
  assert(!gempty_.count(gid));
  if (!--it->second.first)
    gempty_.insert(gid);
  acc.reset();
  assert(acc.nfeatures() == it->second.second.size());
  for (size_t i = 0; i < acc.nfeatures(); i++, acc.bump()) {
    if (unlikely(acc.ismasked()))
      continue;
    it->second.second[i]->remove_value(*models_[i], acc, rng);
  }
  assignments_[eid] = -1;
  return gid;
}

pair<vector<size_t>, vector<float>>
state::score_value(row_accessor &acc, rng_t &rng) const
{
  pair<vector<size_t>, vector<float>> ret;
  const size_t n_empty_groups = gempty_.size();
  assert(n_empty_groups);
  const float empty_group_alpha = alpha_ / float(n_empty_groups);
  size_t count = 0;
  for (auto &group : groups_) {
    float sum = logf(group.second.first ? float(group.second.first) : empty_group_alpha);
    acc.reset();
    for (size_t i = 0; i < acc.nfeatures(); i++, acc.bump()) {
      if (unlikely(acc.ismasked()))
        continue;
      sum += group.second.second[i]->score_value(*models_[i], acc, rng);
    }
    ret.first.push_back(group.first);
    ret.second.push_back(sum);
    count += group.second.first;
  }
  const float lgnorm = logf(float(count) + alpha_);
  for (auto &s : ret.second)
    s -= lgnorm;
  return ret;
}

float
state::score_data(const vector<size_t> &features,
                  rng_t &rng) const
{
  // XXX: out of laziness, we copy
  vector<size_t> fids(features);
  if (fids.empty())
    util::inplace_range(fids, models_.size());
  float sum = 0.;
  for (auto &g : groups_) {
    const auto &gdata = g.second.second;
    for (auto f : fids)
      sum += gdata[f]->score_data(*models_[f], rng);
  }
  return sum;
}

void
state::sample_post_pred(row_accessor &acc,
                        row_mutator &mut,
                        rng_t &rng) const
{
  assert(acc.nfeatures() == mut.nfeatures());
  auto scores = score_value(acc, rng);
  const auto choice = scores.first[util::sample_discrete_log(scores.second, rng)];
  const auto &gdata = groups_.at(choice).second;
  acc.reset(); mut.reset();
  for (size_t i = 0; !acc.end(); acc.bump(), mut.bump(), i++) {
    if (!acc.ismasked()) {
      mut.set(acc);
      continue;
    }
    gdata[i]->sample_value(*models_[i], mut, rng);
  }
}
