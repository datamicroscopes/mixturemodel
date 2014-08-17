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

typedef std::vector<std::shared_ptr<models::group>> group_type;

static inline common::serialized_t
group_type_to_string(const group_type &groups)
{
  io::MixtureModelGroup m;
  for (auto &px : groups)
    m.add_suffstats(px->get_ss());
  return common::util::protobuf_to_string(m);
}

static inline group_type
group_type_from_string(
    const common::serialized_t &s,
    const std::vector<std::shared_ptr<models::hypers>> &models)
{
  common::rng_t rng; // XXX: hack
  io::MixtureModelGroup m;
  common::util::protobuf_from_string(m, s);
  MICROSCOPES_DCHECK((size_t)m.suffstats_size() == models.size(), "sizes do not match");
  group_type g;
  g.reserve(models.size());
  for (size_t i = 0; i < models.size(); i++) {
    g.emplace_back(models[i]->create_group(rng));
    g.back()->set_ss(m.suffstats(i));
  }
  return g;
}

template <template <typename> class GroupManager>
class state {
public:
  typedef typename GroupManager<group_type>::message_type message_type;

  state(const std::vector<std::shared_ptr<models::hypers>> &hypers,
        const GroupManager<group_type> &groups)
    : hypers_(hypers), groups_(groups)
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
    MICROSCOPES_DCHECK(i < hypers_.size(), "invalid feature");
    return hypers_[i]->get_hp();
  }

  inline void
  set_feature_hp(size_t i, const common::hyperparam_bag_t &hp)
  {
    MICROSCOPES_DCHECK(i < hypers_.size(), "invalid feature");
    hypers_[i]->set_hp(hp);
  }

  inline void
  set_feature_hp(size_t i, const models::hypers &m)
  {
    MICROSCOPES_DCHECK(i < hypers_.size(), "invalid feature");
    hypers_[i]->set_hp(m);
  }

  inline common::value_mutator
  get_feature_hp_mutator(size_t i, const std::string &key)
  {
    MICROSCOPES_DCHECK(i < hypers_.size(), "invalid feature");
    return hypers_[i]->get_hp_mutator(key);
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
  inline size_t nfeatures() const { return hypers_.size(); }

  inline size_t groupsize(size_t gid) const { return groups_.groupsize(gid); }
  inline std::vector<size_t> groups() const { return groups_.groups(); }
  inline bool isactivegroup(size_t gid) const { return groups_.isactivegroup(gid); }

  inline void
  add_value(size_t gid,
            size_t eid,
            common::recarray::row_accessor &acc,
            common::rng_t &rng)
  {
    auto &g = groups_.add_value(gid, eid);
    acc.reset();
    MICROSCOPES_DCHECK(acc.nfeatures() == g.size(),
        "nfeatures mismatch");
    const size_t nfeatures = acc.nfeatures();
    for (size_t i = 0; i < nfeatures; i++, acc.bump()) {
      // XXX: currently, multi-dimensional features are all or nothing; if any
      // of the individual values are masked, we treat the whole feature value
      // as masked
      auto value = acc.get();
      if (unlikely(value.anymasked()))
        continue;
      g[i]->add_value(*hypers_[i], value, rng);
    }
  }

  inline size_t
  remove_value(size_t eid,
               common::recarray::row_accessor &acc,
               common::rng_t &rng)
  {
    auto ret = groups_.remove_value(eid);
    auto &g = ret.second;
    acc.reset();
    MICROSCOPES_DCHECK(acc.nfeatures() == g.size(),
        "nfeatures mismatch");
    const size_t nfeatures = acc.nfeatures();
    for (size_t i = 0; i < nfeatures; i++, acc.bump()) {
      // XXX: see note in state::add_value()
      auto value = acc.get();
      if (unlikely(value.anymasked()))
        continue;
      g[i]->remove_value(*hypers_[i], value, rng);
    }
    return ret.first;
  }

  inline std::pair<std::vector<size_t>, std::vector<float>>
  score_value(common::recarray::row_accessor &acc, common::rng_t &rng) const
  {
    std::pair<std::vector<size_t>, std::vector<float>> ret;
    ret.first.reserve(ngroups());
    ret.second.reserve(ngroups());
    inplace_score_value(ret, acc, rng);
    return ret;
  }

  inline void
  inplace_score_value(
    std::pair<std::vector<size_t>, std::vector<float>> &scores,
    common::recarray::row_accessor &acc,
    common::rng_t &rng) const
  {
    MICROSCOPES_DCHECK(acc.nfeatures() == nfeatures(),
        "nfeatures mismatch");

    scores.first.clear();
    scores.second.clear();

    using distributions::fast_log;

    // stash the value_accessors so we don't have to keep
    // reconstructing them in the inner loop below
    std::vector<std::pair<bool, common::value_accessor>> accessors;
    accessors.reserve(nfeatures());
    acc.reset();
    for (size_t i = 0; i < acc.nfeatures(); i++, acc.bump()) {
      auto value = acc.get();
      if (unlikely(value.anymasked()))
        accessors.emplace_back(true, common::value_accessor());
      else
        accessors.emplace_back(false, value);
    }

    float pseudocounts = 0.;
    for (const auto &group : groups_) {
      const float pseudocount =
        groups_.pseudocount(group.first, group.second);
      float sum = fast_log(pseudocount);
      MICROSCOPES_ASSERT(accessors.size() == group.second.data_.size());
      for (size_t i = 0; i < accessors.size(); i++) {
        if (unlikely(accessors[i].first))
          continue;
        sum += group.second.data_[i]->score_value(
            *hypers_[i], accessors[i].second, rng);
      }
      scores.first.push_back(group.first);
      scores.second.push_back(sum);
      pseudocounts += pseudocount;
    }

    const float lgnorm = fast_log(pseudocounts);
    for (auto &s : scores.second)
      s -= lgnorm;
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
      common::util::inplace_range(fids, hypers_.size());
    std::vector<size_t> gids(gs);
    if (gids.empty())
      gids = groups();
    float sum = 0.;
    for (auto gid : gids) {
      const auto &gdata = groups_.group(gid).data_;
      for (auto f : fids)
        sum += gdata[f]->score_data(*hypers_[f], rng);
    }
    return sum;
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
      gdata[i]->sample_value(*hypers_[i], value_mut, rng);
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

  common::serialized_t
  serialize() const
  {
    io::MixtureModelState m;
    for (auto &p : hypers_)
      m.add_hypers(p->get_hp());
    m.set_groups(groups_.serialize(group_type_to_string));
    return common::util::protobuf_to_string(m);
  }

protected:
  std::vector<std::shared_ptr<models::hypers>> hypers_;
  GroupManager<group_type> groups_;
};

// template instantiation
extern template class state<common::fixed_group_manager>;
extern template class state<common::group_manager>;

class model_definition {
public:
  model_definition(
      size_t n,
      const std::vector<std::shared_ptr<models::model>> &models)
    : n_(n), models_(models)
  {
    MICROSCOPES_DCHECK(n > 0, "no entities given");
    MICROSCOPES_DCHECK(models.size() > 0, "no features given");
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

  std::vector<std::shared_ptr<models::hypers>>
  create_hypers() const
  {
    std::vector<std::shared_ptr<models::hypers>> ret;
    ret.reserve(models_.size());
    for (auto &m : models_)
      ret.emplace_back(m->create_hypers());
    return ret;
  }

  inline size_t n() const { return n_; }

  inline const std::vector<std::shared_ptr<models::model>> &
  models() const
  {
    return models_;
  }

  inline size_t
  nmodels() const
  {
    return models().size();
  }

private:
  size_t n_;
  std::vector<std::shared_ptr<models::model>> models_;
};

} // namespace detail

class fixed_model_definition : public detail::model_definition {
public:
  fixed_model_definition(
      size_t n,
      size_t groups,
      const std::vector<std::shared_ptr<models::model>> &models)
    : detail::model_definition(n, models), groups_(groups) {}
  inline size_t groups() const { return groups_; }
private:
  size_t groups_;
};

class model_definition : public detail::model_definition {
public:
  model_definition(
      size_t n,
      const std::vector<std::shared_ptr<models::model>> &models)
    : detail::model_definition(n, models) {}
};

class fixed_state : public detail::state<common::fixed_group_manager> {
public:
  fixed_state(const std::vector<std::shared_ptr<models::hypers>> &hypers,
              const common::fixed_group_manager<detail::group_type> &groups)
    : detail::state<common::fixed_group_manager>(hypers, groups)
  {}

  static std::shared_ptr<fixed_state>
  unsafe_initialize(const fixed_model_definition &def,
                    common::rng_t &rng)
  {
    std::shared_ptr<fixed_state> s = std::make_shared<fixed_state>(
        def.create_hypers(),
        common::fixed_group_manager<detail::group_type>(
          def.n(), def.groups()));
    for (size_t i = 0; i < s->hypers_.size(); i++) {
      auto &gdata = s->groups_.group(i).data_;
      gdata.reserve(s->hypers_.size());
      for (auto &m : s->hypers_)
        gdata.emplace_back(m->create_group(rng));
    }
    return s;
  }

  /**
   * randomly initializes to a valid point in the state space
   *
   * if the assignment vector passed in is empty, generates a random one;
   * otherwise, uses the assignment vector
   */
  static std::shared_ptr<fixed_state>
  initialize(const fixed_model_definition &def,
             const common::hyperparam_bag_t &cluster_init,
             const std::vector<common::hyperparam_bag_t> &feature_inits,
             const std::vector<size_t> &assignments,
             common::recarray::dataview &data,
             common::rng_t &rng)
  {
    auto p = unsafe_initialize(def, rng);
    MICROSCOPES_DCHECK(def.models().size() == feature_inits.size(),
        "init size mismatch");
    MICROSCOPES_DCHECK(def.n() == data.size(),
        "data size mismatch");
    p->set_cluster_hp(cluster_init);
    for (size_t i = 0; i < feature_inits.size(); i++)
      p->set_feature_hp(i, feature_inits[i]);
    std::vector<size_t> assign;
    if (assignments.empty())
      assign = common::util::random_assignment_vector(data.size(), rng);
    else {
      MICROSCOPES_DCHECK(assignments.size() == data.size(),
        "invalid length assignment vector");
      MICROSCOPES_DCHECK(
        *std::max_element(
            assignments.begin(), assignments.end()) < def.groups(),
        "invalid assignment vector");
      assign = assignments;
    }
    data.reset();
    for (size_t i = 0; i < assign.size(); i++, data.next()) {
      auto acc = data.get();
      p->add_value(assign[i], i, acc, rng);
    }
    return p;
  }

  static std::shared_ptr<fixed_state>
  deserialize(const fixed_model_definition &def,
              const common::serialized_t &s)
  {
    std::vector<std::shared_ptr<models::hypers>> hypers;
    hypers.reserve(def.models().size());
    io::MixtureModelState m;
    common::util::protobuf_from_string(m, s);
    MICROSCOPES_DCHECK((size_t)m.hypers_size() == def.models().size(), "inconsistent");
    for (size_t i = 0; i < def.models().size(); i++) {
      auto &p = def.models()[i];
      hypers.emplace_back(p->create_hypers());
      hypers.back()->set_hp(m.hypers(i));
    }
    common::fixed_group_manager<detail::group_type> fg(
        m.groups(), [&hypers](const std::string &s) {
          return detail::group_type_from_string(s, hypers);
        });
    return std::make_shared<fixed_state>(hypers, fg);
  }
};

class state : public detail::state<common::group_manager> {
public:
  state(const std::vector<std::shared_ptr<models::hypers>> &hypers,
        const common::group_manager<detail::group_type> &groups)
    : detail::state<common::group_manager>(hypers, groups)
  {}

  inline size_t
  create_group(common::rng_t &rng)
  {
    auto ret = groups_.create_group();
    auto &gdata = ret.second;
    gdata.reserve(hypers_.size());
    for (auto &m : hypers_)
      gdata.emplace_back(m->create_group(rng));
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

  /**
   * initialized to an **invalid** point in the state space!
   *
   *   (A) no entities assigned
   *   (B) no groups
   *   (C) no hypers initialized
   *
   * useful primarily for testing purposes
   */
  static std::shared_ptr<state>
  unsafe_initialize(const model_definition &def)
  {
    return std::make_shared<state>(
        def.create_hypers(),
        common::group_manager<detail::group_type>(def.n()));
  }

  /**
   * randomly initializes to a valid point in the state space
   */
  static std::shared_ptr<state>
  initialize(const model_definition &def,
             const common::hyperparam_bag_t &cluster_init,
             const std::vector<common::hyperparam_bag_t> &feature_inits,
             const std::vector<size_t> &assignments,
             common::recarray::dataview &data,
             common::rng_t &rng)
  {
    MICROSCOPES_DCHECK(def.models().size() == feature_inits.size(),
        "init size mismatch");
    auto p = unsafe_initialize(def);
    p->set_cluster_hp(cluster_init);
    for (size_t i = 0; i < feature_inits.size(); i++)
      p->set_feature_hp(i, feature_inits[i]);
    std::vector<size_t> assign;
    if (assignments.empty())
      assign = common::util::random_assignment_vector(data.size(), rng);
    else {
      MICROSCOPES_DCHECK(assignments.size() == data.size(),
        "invalid length assignment vector");
      assign = assignments;
    }
    const size_t ngroups = *std::max_element(assign.begin(), assign.end()) + 1;
    for (size_t i = 0; i < ngroups; i++)
      p->create_group(rng);
    data.reset();
    for (size_t i = 0; i < assign.size(); i++, data.next()) {
      auto acc = data.get();
      p->add_value(assign[i], i, acc, rng);
    }
    return p;
  }

  static std::shared_ptr<state>
  deserialize(const model_definition &def,
              const common::serialized_t &s)
  {
    std::vector<std::shared_ptr<models::hypers>> hypers;
    hypers.reserve(def.models().size());
    io::MixtureModelState m;
    common::util::protobuf_from_string(m, s);
    MICROSCOPES_DCHECK((size_t)m.hypers_size() == def.models().size(), "inconsistent");
    for (size_t i = 0; i < def.models().size(); i++) {
      auto &p = def.models()[i];
      hypers.emplace_back(p->create_hypers());
      hypers.back()->set_hp(m.hypers(i));
    }
    common::group_manager<detail::group_type> g(
        m.groups(),
        [&hypers](const std::string &s) {
          return detail::group_type_from_string(s, hypers);
        });
    return std::make_shared<state>(hypers, g);
  }

};

namespace detail {

template <typename T, typename Base>
class model : public Base {
public:
  model(const std::shared_ptr<T> &impl,
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
                   const models::hypers &proto) override
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

  void
  inplace_score_value(
      std::pair<std::vector<size_t>, std::vector<float>> &scores,
      size_t eid,
      common::rng_t &rng) const override
  {
    common::recarray::row_accessor acc = data_->get(eid);
    impl_->inplace_score_value(scores, acc, rng);
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

  inline const std::shared_ptr<T> & state() const { return impl_; }

protected:
  std::shared_ptr<T> impl_;
  std::shared_ptr<common::recarray::dataview> data_;
};

extern template class model<
  mixture::fixed_state,
  common::fixed_entity_based_state_object
>;

extern template class model<
  mixture::state,
  common::entity_based_state_object
>;

} // namespace detail

class fixed_model :
  public detail::model<
      fixed_state,
      common::fixed_entity_based_state_object>
{
public:
  fixed_model(
      const std::shared_ptr<fixed_state> &impl,
      const std::shared_ptr<common::recarray::dataview> &data)
    : detail::model<
        fixed_state,
        common::fixed_entity_based_state_object>(impl, data)
  {}
};

class model :
  public detail::model<
      mixture::state,
      common::entity_based_state_object>
{
public:
  model(
      const std::shared_ptr<mixture::state> &impl,
      const std::shared_ptr<common::recarray::dataview> &data)
    : detail::model<
        mixture::state,
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
