#pragma once

#include <microscopes/models/base.hpp>
#include <microscopes/common/typedefs.hpp>
#include <microscopes/io/schema.pb.h>

#include <cassert>
#include <cmath>
#include <vector>
#include <set>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <utility>

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

  common::hyperparam_bag_t
  get_hp() const
  {
    message_type m;
    m.set_alpha(alpha_);
    std::ostringstream out;
    m.SerializeToOstream(&out);
    return out.str();
  }

  void
  set_hp(const common::hyperparam_bag_t &hp)
  {
    std::istringstream inp(hp);
    message_type m;
    m.ParseFromIstream(&inp);
    alpha_ = m.alpha();
  }

  common::hyperparam_bag_t
  get_feature_hp(size_t i) const
  {
    return models_[i]->get_hp();
  }

  void
  set_feature_hp(size_t i, const common::hyperparam_bag_t &hp)
  {
    models_[i]->set_hp(hp);
  }

  common::suffstats_bag_t
  get_suff_stats(size_t gid, size_t fid) const
  {
    const auto it = groups_.find(gid);
    assert(it != groups_.end());
    assert(fid < it->second.second.size());
    return it->second.second[fid]->get_ss();
  }

  void
  set_suff_stats(size_t gid, size_t fid, const common::suffstats_bag_t &ss)
  {
    const auto it = groups_.find(gid);
    assert(it != groups_.end());
    assert(fid < it->second.second.size());
    it->second.second[fid]->set_ss(ss);
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

  inline size_t
  groupsize(size_t gid) const
  {
    const auto it = groups_.find(gid);
    assert(it != groups_.end());
    return it->second.first;
  }

  inline std::vector<size_t>
  groups() const
  {
    std::vector<size_t> ret;
    ret.reserve(ngroups());
    for (auto &g : groups_)
      ret.push_back(g.first);
    return ret;
  }

  size_t create_group(common::rng_t &rng);
  void delete_group(size_t gid);

  void add_value(size_t gid, const common::dataview &view, common::rng_t &rng);
  void add_value(size_t gid, size_t eid, common::row_accessor &acc, common::rng_t &rng);

  size_t remove_value(const common::dataview &view, common::rng_t &rng);
  size_t remove_value(size_t eid, common::row_accessor &acc, common::rng_t &rng);

  std::pair<std::vector<size_t>, std::vector<float>>
  score_value(common::row_accessor &acc, common::rng_t &rng) const;

  float score_data(const std::vector<size_t> &features, const std::vector<size_t> &groups, common::rng_t &rng) const;


  // XXX: helper function, move to outer mixturemodel once we
  // abstract better
  void ensure_k_empty_groups(size_t k, common::rng_t &rng);

  // XXX: also doesn't belong here
  std::vector< runtime_type_info >
  get_runtime_type_info() const;


  // XXX: we assume the caller has taken care to set the groups correctly!
  void sample_post_pred(common::row_accessor &acc, common::row_mutator &mut, common::rng_t &rng) const;

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
