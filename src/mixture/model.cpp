#include <microscopes/mixture/model.hpp>
#include <microscopes/common/macros.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/common/util.hpp>
#include <distributions/special.hpp>

#include <iostream>

using namespace std;
using namespace distributions;
using namespace microscopes::common;
using namespace microscopes::common::recarray;
using namespace microscopes::models;
using namespace microscopes::mixture;

size_t
state::groupsize(size_t gid) const
{
  const auto it = groups_.find(gid);
  MICROSCOPES_ASSERT(it != groups_.end());
  return it->second.first;
}

vector<size_t>
state::groups() const
{
  vector<size_t> ret;
  ret.reserve(ngroups());
  for (auto &g : groups_)
    ret.push_back(g.first);
  return move(ret);
}

size_t
state::create_group(rng_t &rng)
{
  vector<shared_ptr<feature_group>> gdata;
  gdata.reserve(models_.size());
  for (auto &m : models_)
    gdata.emplace_back(m->create_feature_group(rng));
  const size_t gid = gcount_++;
  groups_[gid] = move(make_pair(0, move(gdata)));
  MICROSCOPES_ASSERT(!gempty_.count(gid));
  gempty_.insert(gid);
  return gid;
}

void
state::delete_group(size_t gid)
{
  auto it = groups_.find(gid);
  MICROSCOPES_ASSERT(it != groups_.end());
  MICROSCOPES_ASSERT(!it->second.first);
  MICROSCOPES_ASSERT(gempty_.count(gid));
  groups_.erase(it);
  gempty_.erase(gid);
  gremoved_++;
}

void
state::ensure_k_empty_groups(size_t k, bool resample, rng_t &rng)
{
  if (resample) {
    // delete all empty groups
    const vector<size_t> egids(gempty_.begin(), gempty_.end());
    for (auto egid : egids)
      delete_group(egid);
  }
  const size_t esize = empty_groups().size();
  if (esize == k)
    return;
  else if (esize > k) {
    // set iterators do not support iter + size_type
    auto it = gempty_.cbegin();
    for (size_t i = 0; i < (esize-k); ++i, ++it)
      ;
    const vector<size_t> egids(it, gempty_.cend());
    for (auto egid : egids)
      delete_group(egid);
  } else {
    for (size_t i = 0; i < (k-esize); i++)
      create_group(rng);
  }
  MICROSCOPES_ASSERT( empty_groups().size() == k );
}

vector<runtime_type>
state::get_runtime_types() const
{
  vector<runtime_type> ret;
  ret.reserve(models_.size());
  for (const auto &m : models_)
    ret.push_back(m->get_runtime_type());
  return ret;
}

void
state::add_value(size_t gid, const dataview &view, rng_t &rng)
{
  MICROSCOPES_ASSERT(view.size() == assignments_.size());
  row_accessor acc = view.get();
  const size_t eid = view.index();
  add_value(gid, eid, acc, rng);
}

void
state::add_value(size_t gid, size_t eid, row_accessor &acc, rng_t &rng)
{
  MICROSCOPES_ASSERT(assignments_.at(eid) == -1);
  auto it = groups_.find(gid);
  MICROSCOPES_ASSERT(it != groups_.end());
  if (!it->second.first++) {
    MICROSCOPES_ASSERT(gempty_.count(gid));
    gempty_.erase(gid);
    MICROSCOPES_ASSERT(!gempty_.count(gid));
  } else {
    MICROSCOPES_ASSERT(!gempty_.count(gid));
  }
  acc.reset();
  MICROSCOPES_ASSERT(acc.nfeatures() == it->second.second.size());
  for (size_t i = 0; i < acc.nfeatures(); i++, acc.bump()) {
    // XXX: currently, multi-dimensional features are all or nothing; if any of
    // the individual values are masked, we treat the whole feature value as
    // masked
    if (unlikely(acc.anymasked()))
      continue;
    it->second.second[i]->add_value(*models_[i], acc.get(), rng);
  }
  assignments_[eid] = gid;
}

size_t
state::remove_value(const dataview &view, rng_t &rng)
{
  MICROSCOPES_ASSERT(view.size() == assignments_.size());
  row_accessor acc = view.get();
  const size_t eid = view.index();
  return remove_value(eid, acc, rng);
}

size_t
state::remove_value(size_t eid, row_accessor &acc, rng_t &rng)
{
  MICROSCOPES_ASSERT(assignments_.at(eid) != -1);
  const size_t gid = assignments_[eid];
  auto it = groups_.find(gid);
  MICROSCOPES_ASSERT(it != groups_.end());
  MICROSCOPES_ASSERT(!gempty_.count(gid));
  if (!--it->second.first)
    gempty_.insert(gid);
  acc.reset();
  MICROSCOPES_ASSERT(acc.nfeatures() == it->second.second.size());
  for (size_t i = 0; i < acc.nfeatures(); i++, acc.bump()) {
    // XXX: see note in state::add_value()
    if (unlikely(acc.anymasked()))
      continue;
    it->second.second[i]->remove_value(*models_[i], acc.get(), rng);
  }
  assignments_[eid] = -1;
  return gid;
}

pair<vector<size_t>, vector<float>>
state::score_value(row_accessor &acc, rng_t &rng) const
{
  pair<vector<size_t>, vector<float>> ret;
  ret.first.reserve(groups_.size());
  ret.second.reserve(groups_.size());
  const size_t n_empty_groups = gempty_.size();
  MICROSCOPES_ASSERT(n_empty_groups);
  const float empty_group_alpha = alpha_ / float(n_empty_groups);
  size_t count = 0;
  for (auto &group : groups_) {
    float sum = fast_log(group.second.first ? float(group.second.first) : empty_group_alpha);
    acc.reset();
    for (size_t i = 0; i < acc.nfeatures(); i++, acc.bump()) {
      if (unlikely(acc.anymasked()))
        continue;
      sum += group.second.second[i]->score_value(*models_[i], acc.get(), rng);
    }
    ret.first.push_back(group.first);
    ret.second.push_back(sum);
    count += group.second.first;
  }
  const float lgnorm = fast_log(float(count) + alpha_);
  for (auto &s : ret.second)
    s -= lgnorm;
  return ret;
}

float
state::score_data(const vector<size_t> &fs,
                  const vector<size_t> &gs,
                  rng_t &rng) const
{
  // XXX: out of laziness, we copy
  vector<size_t> fids(fs);
  if (fids.empty())
    util::inplace_range(fids, models_.size());
  vector<size_t> gids(gs);
  if (gids.empty())
    gids = groups();
  float sum = 0.;
  for (auto gid : gids) {
    const auto &gdata = groups_.at(gid).second;
    for (auto f : fids)
      sum += gdata[f]->score_data(*models_[f], rng);
  }
  return sum;
}

size_t
state::sample_post_pred(row_accessor &acc,
                        row_mutator &mut,
                        rng_t &rng) const
{
  MICROSCOPES_ASSERT(acc.nfeatures() == mut.nfeatures());
  auto scores = score_value(acc, rng);
  const auto choice = scores.first[util::sample_discrete_log(scores.second, rng)];
  const auto &gdata = groups_.at(choice).second;

  //cout << "sample_post_pred():" << endl
  //     << "  choice=" << choice << endl
  //     << "  probs=" << scores.second << endl
  //     << "  acc= " << acc.debug_str() << endl;

  acc.reset();
  mut.reset();
  for (size_t i = 0; !acc.end(); acc.bump(), mut.bump(), i++) {
    if (!acc.anymasked()) {
      mut.set(acc);
      continue;
    }
    auto value_mut = mut.set();
    gdata[i]->sample_value(*models_[i], value_mut, rng);
  }

  return choice;
}

float
state::score_assignment() const
{
  map<size_t, size_t> counts;
  MICROSCOPES_DCHECK(assignments_[0] != -1, "assignment is not valid");
  counts[assignments_[0]] = 1;
  float sum = 0.;
  for (size_t i = 1; i < assignments_.size(); i++) {
    const ssize_t gid = assignments_[i];
    MICROSCOPES_DCHECK(gid != -1, "assignment is not valid");
    const auto it = counts.find(gid);
    const bool found = (it != counts.end());
    const float numer = (!found) ? alpha_ : it->second;
    const float denom = float(i) + alpha_;
    sum += fast_log(numer / denom);
    if (found)
      it->second++;
    else
      counts[gid] = 1;
  }
  return sum;
}

void
state::dcheck_consistency() const
{
  MICROSCOPES_DCHECK(gcount_ >= gremoved_, "created is not >= removed");
  MICROSCOPES_DCHECK(alpha_ > 0.0, "cluster HP <= 0.0");

  // check the assignments are all valid
  map<size_t, size_t> counts;
  for (size_t i = 0; i < assignments_.size(); i++) {
    if (assignments_[i] == -1)
      continue;
    MICROSCOPES_DCHECK(assignments_[i] >= 0, "invalid negative assignment found");
    MICROSCOPES_DCHECK(!gempty_.count(assignments_[i]), "assigned element in empty group");
    MICROSCOPES_DCHECK(groups_.find(assignments_[i]) != groups_.end(), "assigned to non-existent group");
    counts[assignments_[i]]++;
  }

  // every group in gempty_ should appear in groups_, but empty
  for (auto g : gempty_) {
    const auto it = groups_.find(g);
    MICROSCOPES_DCHECK(it != groups_.end(), "non-existent group in empty groups list");
    MICROSCOPES_DCHECK(it->second.first == 0, "empty group is not empty");
    // XXX: need a way to tell suff stats are actually empty!
  }

  for (auto g : groups_) {
    if (!g.second.first) {
      MICROSCOPES_DCHECK(gempty_.count(g.first), "empty group not accounted for in gempty_");
      MICROSCOPES_DCHECK(counts.find(g.first) == counts.end(), "empty group not empty");
    } else {
      MICROSCOPES_DCHECK(!gempty_.count(g.first), "non-empty group found in gempty_");
      MICROSCOPES_DCHECK(counts.at(g.first) == g.second.first, "assignments disagree");
    }
  }
}
