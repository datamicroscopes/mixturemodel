#pragma once

#include <microscopes/models/base.hpp>
#include <microscopes/common/entity_state.hpp>
#include <microscopes/common/group_manager.hpp>
#include <microscopes/common/recarray/dataview.hpp>
#include <microscopes/common/util.hpp>
#include <microscopes/common/typedefs.hpp>
#include <microscopes/common/assert.hpp>
#include <microscopes/io/schema.pb.h>
#include <distributions/special.hpp>

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

namespace detail {

typedef std::vector<std::shared_ptr<models::feature_group>> group_type;

template <template <typename> class GroupManager>
class state {
public:
  typedef typename GroupManager<group_type>::message_type message_type;

  state(const std::vector<std::shared_ptr<models::model>> &models,
        const GroupManager<group_type> &groups)
    : models_(models),
      groups_(groups)
  {}

  inline common::hyperparam_bag_t
  get_cluster_hp() const
  {
    return groups_.get_hp();
  }

  inline void
  set_cluster_hp(const common::hyperparam_bag_t &hp)
  {
    groups_.set_hp(hp);
  }

  inline common::value_mutator
  get_cluster_hp_mutator(const std::string &key)
  {
    return groups_.get_hp_mutator(key);
  }

  inline common::hyperparam_bag_t
  get_feature_hp(size_t i) const
  {
    MICROSCOPES_DCHECK(i < models_.size(), "invalid feature");
    return models_[i]->get_hp();
  }

  inline void
  set_feature_hp(size_t i, const common::hyperparam_bag_t &hp)
  {
    MICROSCOPES_DCHECK(i < models_.size(), "invalid feature");
    models_[i]->set_hp(hp);
  }

  inline void
  set_feature_hp(size_t i, const models::model &m)
  {
    MICROSCOPES_DCHECK(i < models_.size(), "invalid feature");
    models_[i]->set_hp(m);
  }

  inline common::value_mutator
  get_feature_hp_mutator(size_t i, const std::string &key)
  {
    MICROSCOPES_DCHECK(i < models_.size(), "invalid feature");
    return models_[i]->get_hp_mutator(key);
  }

  inline common::suffstats_bag_t
  get_suffstats(size_t gid, size_t fid) const
  {
    const auto &g = groups_.group(gid).data_;
    MICROSCOPES_DCHECK(fid < g.size(), "invalid feature");
    return g[fid]->get_ss();
  }

  inline void
  set_suffstats(size_t gid, size_t fid, const common::suffstats_bag_t &ss)
  {
    const auto &g = groups_.group(gid).data_;
    MICROSCOPES_DCHECK(fid < g.size(), "invalid feature");
    g[fid]->set_ss(ss);
  }

  inline common::value_mutator
  get_suffstats_mutator(size_t gid, size_t fid, const std::string &key)
  {
    const auto &g = groups_.group(gid).data_;
    MICROSCOPES_DCHECK(fid < g.size(), "invalid feature");
    return g[fid]->get_ss_mutator(key);
  }

  inline const std::vector<ssize_t> &
  assignments() const
  {
    return groups_.assignments();
  }

  inline size_t nentities() const { return groups_.nentities(); }
  inline size_t ngroups() const { return groups_.ngroups(); }
  inline size_t nfeatures() const { return models_.size(); }

  inline size_t groupsize(size_t gid) const { return groups_.groupsize(gid); }
  inline std::vector<size_t> groups() const { return groups_.groups(); }

  inline void
  add_value(size_t gid, const common::recarray::dataview &view, common::rng_t &rng)
  {
    MICROSCOPES_DCHECK(view.size() == nentities(), "invalid view");
    common::recarray::row_accessor acc = view.get();
    const size_t eid = view.index();
    add_value(gid, eid, acc, rng);
  }

  inline void
  add_value(size_t gid, size_t eid, common::recarray::row_accessor &acc, common::rng_t &rng)
  {
    auto &g = groups_.add_value(gid, eid);
    acc.reset();
    MICROSCOPES_ASSERT(acc.nfeatures() == g.size());
    for (size_t i = 0; i < acc.nfeatures(); i++, acc.bump()) {
      // XXX: currently, multi-dimensional features are all or nothing; if any of
      // the individual values are masked, we treat the whole feature value as
      // masked
      if (unlikely(acc.anymasked()))
        continue;
      g[i]->add_value(*models_[i], acc.get(), rng);
    }
  }

  inline size_t
  remove_value(const common::recarray::dataview &view, common::rng_t &rng)
  {
    MICROSCOPES_DCHECK(view.size() == nentities(), "invalid view");
    common::recarray::row_accessor acc = view.get();
    const size_t eid = view.index();
    return remove_value(eid, acc, rng);
  }

  inline size_t
  remove_value(size_t eid, common::recarray::row_accessor &acc, common::rng_t &rng)
  {
    auto ret = groups_.remove_value(eid);
    auto &g = ret.second;
    acc.reset();
    MICROSCOPES_ASSERT(acc.nfeatures() == g.size());
    for (size_t i = 0; i < acc.nfeatures(); i++, acc.bump()) {
      // XXX: see note in state::add_value()
      if (unlikely(acc.anymasked()))
        continue;
      g[i]->remove_value(*models_[i], acc.get(), rng);
    }
    return ret.first;
  }

  std::pair<std::vector<size_t>, std::vector<float>>
  score_value(common::recarray::row_accessor &acc, common::rng_t &rng) const
  {
    using distributions::fast_log;
    std::pair<std::vector<size_t>, std::vector<float>> ret;
    ret.first.reserve(ngroups());
    ret.second.reserve(ngroups());
    float pseudocounts = 0.;
    for (auto &group : groups_) {
      const float pseudocount = groups_.pseudocount(group.first, group.second);
      float sum = fast_log(pseudocount);
      acc.reset();
      MICROSCOPES_ASSERT(acc.nfeatures() == group.second.data_.size());
      for (size_t i = 0; i < acc.nfeatures(); i++, acc.bump()) {
        if (unlikely(acc.anymasked()))
          continue;
        sum += group.second.data_[i]->score_value(*models_[i], acc.get(), rng);
      }
      ret.first.push_back(group.first);
      ret.second.push_back(sum);
      pseudocounts += pseudocount;
    }
    const float lgnorm = fast_log(pseudocounts);
    for (auto &s : ret.second)
      s -= lgnorm;
    return ret;
  }

  // accumulate (sum) score_data over the suff-stats of the cartesian-product
  // of [features] x [groups]
  float
  score_data(const std::vector<size_t> &fs,
             const std::vector<size_t> &gs,
             common::rng_t &rng) const
  {
    // XXX: out of laziness, we copy
    std::vector<size_t> fids(fs);
    if (fids.empty())
      common::util::inplace_range(fids, models_.size());
    std::vector<size_t> gids(gs);
    if (gids.empty())
      gids = groups();
    float sum = 0.;
    for (auto gid : gids) {
      const auto &gdata = groups_.group(gid).data_;
      for (auto f : fids)
        sum += gdata[f]->score_data(*models_[f], rng);
    }
    return sum;
  }

  std::vector<common::runtime_type>
  get_runtime_types() const
  {
    std::vector<common::runtime_type> ret;
    ret.reserve(models_.size());
    for (const auto &m : models_)
      ret.push_back(m->get_runtime_type());
    return ret;
  }

  // XXX: we assume the caller has taken care to set the groups correctly!
  size_t
  sample_post_pred(common::recarray::row_accessor &acc,
                   common::recarray::row_mutator &mut,
                   common::rng_t &rng) const
  {
    MICROSCOPES_DCHECK(acc.nfeatures() == mut.nfeatures(),
        "nfeatures not the same");
    auto scores = score_value(acc, rng);
    const auto choice =
      scores.first[common::util::sample_discrete_log(scores.second, rng)];
    const auto &gdata = groups_.group(choice).data_;

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

  inline float score_assignment() const { return groups_.score_assignment(); }

  inline float
  score_joint(common::rng_t &rng) const
  {
    const std::vector<size_t> empty;
    return score_assignment() + score_data(empty, empty, rng);
  }

  // for debugging purposes
  void
  dcheck_consistency() const
  {
    // XXX: implement me
  }

protected:
  std::vector<std::shared_ptr<models::model>> models_;
  GroupManager<group_type> groups_;
};

// template instantiation
extern template class state<common::fixed_group_manager>;
extern template class state<common::group_manager>;

} // namespace detail

class fixed_state : public detail::state<common::fixed_group_manager> {
public:
  fixed_state(size_t n, size_t k,
              const std::vector<std::shared_ptr<models::model>> &models)
    : detail::state<common::fixed_group_manager>(
        models,
        common::fixed_group_manager<detail::group_type>(n, k))
  {}
};

class state : public detail::state<common::group_manager> {
public:
  state(size_t n, const std::vector<std::shared_ptr<models::model>> &models)
    : detail::state<common::group_manager>(
        models,
        common::group_manager<detail::group_type>(n))
  {}

  inline size_t
  create_group(common::rng_t &rng)
  {
    auto ret = groups_.create_group();
    auto &gdata = ret.second;
    gdata.reserve(models_.size());
    for (auto &m : models_)
      gdata.emplace_back(m->create_feature_group(rng));
    return ret.first;
  }

  inline void
  delete_group(size_t gid)
  {
    groups_.delete_group(gid);
  }

  inline const std::set<size_t> &
  empty_groups() const
  {
    return groups_.empty_groups();
  }

  // XXX: helper function, move to outer mixturemodel once we
  // abstract better
  void
  ensure_k_empty_groups(size_t k, bool resample, common::rng_t &rng)
  {
    if (resample) {
      // delete all empty groups
      const std::vector<size_t> egids(
          empty_groups().begin(),
          empty_groups().end());
      for (auto egid : egids)
        groups_.delete_group(egid);
    }
    const size_t esize = empty_groups().size();
    if (esize == k)
      return;
    else if (esize > k) {
      // set iterators do not support iter + size_type
      auto it = empty_groups().cbegin();
      for (size_t i = 0; i < (esize-k); ++i, ++it)
        ;
      const std::vector<size_t> egids(it, empty_groups().cend());
      for (auto egid : egids)
        delete_group(egid);
    } else {
      for (size_t i = 0; i < (k-esize); i++)
        create_group(rng);
    }
    MICROSCOPES_ASSERT( empty_groups().size() == k );
  }

};

namespace detail {

template <typename T, typename Base>
class bound_state : public Base {
public:
  bound_state(
      const std::shared_ptr<T> &impl,
      const std::shared_ptr<common::recarray::dataview> &data)
    : impl_(impl), data_(data) {}

  size_t nentities() const override { return impl_->nentities(); }
  size_t ngroups() const override { return impl_->ngroups(); }
  size_t ncomponents() const override { return impl_->nfeatures(); }

  std::vector<ssize_t> assignments() const override { return impl_->assignments(); }
  std::vector<size_t> groups() const override { return impl_->groups(); }

  size_t groupsize(size_t gid) const override { return impl_->groupsize(gid); }

  common::hyperparam_bag_t
  get_cluster_hp() const override
  {
    return impl_->get_cluster_hp();
  }

  void
  set_cluster_hp(const common::hyperparam_bag_t &hp) override
  {
    impl_->set_cluster_hp(hp);
  }

  common::value_mutator
  get_cluster_hp_mutator(const std::string &key) override
  {
    return impl_->get_cluster_hp_mutator(key);
  }

  common::hyperparam_bag_t
  get_component_hp(size_t component) const override
  {
    return impl_->get_feature_hp(component);
  }

  void
  set_component_hp(size_t component,
                   const common::hyperparam_bag_t &hp) override
  {
    impl_->set_feature_hp(component, hp);
  }

  void
  set_component_hp(size_t component,
                   const models::model &proto) override
  {
    impl_->set_feature_hp(component, proto);
  }

  common::value_mutator
  get_component_hp_mutator(
      size_t component, const std::string &key) override
  {
    return impl_->get_feature_hp_mutator(component, key);
  }

  std::vector<common::ident_t>
  suffstats_identifiers(size_t component) const override
  {
    return impl_->groups();
  }

  common::suffstats_bag_t
  get_suffstats(size_t component, common::ident_t id) const override
  {
    return impl_->get_suffstats(id, component);
  }

  void
  set_suffstats(
      size_t component, common::ident_t id,
      const common::suffstats_bag_t &ss) override
  {
    impl_->set_suffstats(id, component, ss);
  }

  common::value_mutator
  get_suffstats_mutator(
      size_t component,
      common::ident_t id, const std::string &key) override
  {
    return impl_->get_suffstats_mutator(id, component, key);
  }

  void
  add_value(
      size_t gid, size_t eid, common::rng_t &rng) override
  {
    common::recarray::row_accessor acc = data_->get(eid);
    impl_->add_value(gid, eid, acc, rng);
  }

  size_t
  remove_value(size_t eid, common::rng_t &rng) override
  {
    common::recarray::row_accessor acc = data_->get(eid);
    return impl_->remove_value(eid, acc, rng);
  }

  std::pair<std::vector<size_t>, std::vector<float>>
  score_value(size_t eid, common::rng_t &rng) const override
  {
    common::recarray::row_accessor acc = data_->get(eid);
    return impl_->score_value(acc, rng);
  }

  float score_assignment()
  const override
  {
    return impl_->score_assignment();
  }

  float
  score_likelihood(
      size_t component,
      common::ident_t id,
      common::rng_t &rng) const override
  {
    return impl_->score_data({component}, {id}, rng);
  }

  float
  score_likelihood(
      size_t component, common::rng_t &rng) const override
  {
    return impl_->score_data({component}, {}, rng);
  }

protected:
  std::shared_ptr<T> impl_;
  std::shared_ptr<common::recarray::dataview> data_;
};

extern template class bound_state<
  mixture::fixed_state,
  common::fixed_entity_based_state_object
>;

extern template class bound_state<
  mixture::state,
  common::entity_based_state_object
>;

} // namespace detail

class bound_fixed_state :
  public detail::bound_state<
      fixed_state,
      common::fixed_entity_based_state_object>
{
public:
  bound_fixed_state(
      const std::shared_ptr<fixed_state> &impl,
      const std::shared_ptr<common::recarray::dataview> &data)
    : detail::bound_state<
        fixed_state,
        common::fixed_entity_based_state_object>(impl, data)
  {}
};

class bound_state :
  public detail::bound_state<
      state,
      common::entity_based_state_object>
{
public:
  bound_state(
      const std::shared_ptr<state> &impl,
      const std::shared_ptr<common::recarray::dataview> &data)
    : detail::bound_state<
        state,
        common::entity_based_state_object>(impl, data)
  {}

  std::vector<size_t>
  empty_groups() const override
  {
    const auto &egs = impl_->empty_groups();
    return std::vector<size_t>(egs.begin(), egs.end());
  }

  size_t create_group(common::rng_t &rng) override { return impl_->create_group(rng); }
  void delete_group(size_t gid) override { impl_->delete_group(gid); }
};

} // namespace mixture
} // namespace microscopes
