#pragma once

#include <microscopes/models/base.hpp>
#include <microscopes/common/recarray/dataview.hpp>
#include <microscopes/common/typedefs.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/io/schema.pb.h>

#include <cmath>
#include <vector>
#include <set>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <utility>
#include <stdexcept>

namespace microscopes {
namespace mixture {

class state {
public:

  typedef io::CRP message_type;

  state(size_t n, const std::vector<std::shared_ptr<models::model>> &models)
    : alpha_(),
      gcount_(),
      gremoved_(),
      gempty_(),
      assignments_(n, -1),
      models_(models),
      groups_()
  {
  }

  inline common::hyperparam_bag_t
  get_hp() const
  {
    message_type m;
    m.set_alpha(alpha_);
    std::ostringstream out;
    m.SerializeToOstream(&out);
    return out.str();
  }

  inline void
  set_hp(const common::hyperparam_bag_t &hp)
  {
    std::istringstream inp(hp);
    message_type m;
    m.ParseFromIstream(&inp);
    alpha_ = m.alpha();
  }

  inline void *
  get_cluster_hp_raw_ptr(const std::string &key)
  {
    if (key == "alpha")
      return &alpha_;
    throw std::runtime_error("unknown key: " + key);
  }

  inline common::hyperparam_bag_t
  get_feature_hp(size_t i) const
  {
    return models_[i]->get_hp();
  }

  inline void
  set_feature_hp(size_t i, const common::hyperparam_bag_t &hp)
  {
    models_[i]->set_hp(hp);
  }

  inline void
  set_feature_hp(size_t i, const models::model &m)
  {
    models_[i]->set_hp(m);
  }

  inline void *
  get_feature_hp_raw_ptr(size_t i, const std::string &key)
  {
    return models_[i]->get_hp_raw_ptr(key);
  }

  inline common::suffstats_bag_t
  get_suff_stats(size_t gid, size_t fid) const
  {
    const auto it = groups_.find(gid);
    MICROSCOPES_ASSERT(it != groups_.end());
    MICROSCOPES_ASSERT(fid < it->second.second.size());
    return it->second.second[fid]->get_ss();
  }

  inline void
  set_suff_stats(size_t gid, size_t fid, const common::suffstats_bag_t &ss)
  {
    const auto it = groups_.find(gid);
    MICROSCOPES_ASSERT(it != groups_.end());
    MICROSCOPES_ASSERT(fid < it->second.second.size());
    it->second.second[fid]->set_ss(ss);
  }

  inline void *
  get_suff_stats_raw_ptr(size_t gid, size_t fid, const std::string &key)
  {
    const auto it = groups_.find(gid);
    MICROSCOPES_ASSERT(it != groups_.end());
    MICROSCOPES_ASSERT(fid < it->second.second.size());
    return it->second.second[fid]->get_ss_raw_ptr(key);
  }

  inline const std::vector<ssize_t> &
  assignments() const
  {
    return assignments_;
  }

  inline const std::set<size_t> &
  empty_groups() const
  {
    return gempty_;
  }

  inline size_t nentities() const { return assignments_.size(); }
  inline size_t ngroups() const { return groups_.size(); }

  size_t groupsize(size_t gid) const;
  std::vector<size_t> groups() const;

  size_t create_group(common::rng_t &rng);
  void delete_group(size_t gid);

  void add_value(size_t gid, const common::recarray::dataview &view, common::rng_t &rng);
  void add_value(size_t gid, size_t eid, common::recarray::row_accessor &acc, common::rng_t &rng);

  size_t remove_value(const common::recarray::dataview &view, common::rng_t &rng);
  size_t remove_value(size_t eid, common::recarray::row_accessor &acc, common::rng_t &rng);

  std::pair<std::vector<size_t>, std::vector<float>>
  score_value(common::recarray::row_accessor &acc, common::rng_t &rng) const;

  // accumulate (sum) score_data over the suff-stats of the cartesian-product
  // of [features] x [groups]
  float score_data(
      const std::vector<size_t> &features,
      const std::vector<size_t> &groups,
      common::rng_t &rng) const;

  // XXX: helper function, move to outer mixturemodel once we
  // abstract better
  void ensure_k_empty_groups(size_t k, bool resample, common::rng_t &rng);

  // XXX: also doesn't belong here
  std::vector< common::runtime_type >
  get_runtime_types() const;

  // XXX: we assume the caller has taken care to set the groups correctly!
  size_t sample_post_pred(common::recarray::row_accessor &acc, common::recarray::row_mutator &mut, common::rng_t &rng) const;

  float score_assignment() const;

  inline float
  score_joint(common::rng_t &rng) const
  {
    const std::vector<size_t> empty;
    return score_assignment() + score_data(empty, empty, rng);
  }

  // for debugging purposes
  void dcheck_consistency() const;

  // random statistics
  inline size_t groups_created() const { return gcount_; }
  inline size_t groups_removed() const { return gremoved_; }

private:
  float alpha_;
  size_t gcount_;
  size_t gremoved_;
  std::set<size_t> gempty_;
  std::vector<ssize_t> assignments_;
  std::vector<std::shared_ptr<models::model>> models_;
  std::map<size_t, std::pair<size_t, std::vector<std::shared_ptr<models::feature_group>>>> groups_;
};

} // namespace mixture
} // namespace microscopes
