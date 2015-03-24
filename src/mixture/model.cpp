#include <microscopes/mixture/model.hpp>

namespace microscopes {
namespace mixture {
namespace detail {

template class state<common::group_manager>;
template class model<
  mixture::state,
  common::entity_based_state_object
>;

} // namespace detail
} // namespace mixture
} // namespace microscopes

//void
//state::dcheck_consistency() const
//{
//  MICROSCOPES_DCHECK(gcount_ >= gremoved_, "created is not >= removed");
//  MICROSCOPES_DCHECK(alpha_ > 0.0, "cluster HP <= 0.0");
//
//  // check the assignments are all valid
//  map<size_t, size_t> counts;
//  for (size_t i = 0; i < assignments_.size(); i++) {
//    if (assignments_[i] == -1)
//      continue;
//    MICROSCOPES_DCHECK(assignments_[i] >= 0, "invalid negative assignment found");
//    MICROSCOPES_DCHECK(!gempty_.count(assignments_[i]), "assigned element in empty group");
//    MICROSCOPES_DCHECK(groups_.find(assignments_[i]) != groups_.end(), "assigned to non-existent group");
//    counts[assignments_[i]]++;
//  }
//
//  // every group in gempty_ should appear in groups_, but empty
//  for (auto g : gempty_) {
//    const auto it = groups_.find(g);
//    MICROSCOPES_DCHECK(it != groups_.end(), "non-existent group in empty groups list");
//    MICROSCOPES_DCHECK(it->second.first == 0, "empty group is not empty");
//    // XXX: need a way to tell suff stats are actually empty!
//  }
//
//  for (auto g : groups_) {
//    if (!g.second.first) {
//      MICROSCOPES_DCHECK(gempty_.count(g.first), "empty group not accounted for in gempty_");
//      MICROSCOPES_DCHECK(counts.find(g.first) == counts.end(), "empty group not empty");
//    } else {
//      MICROSCOPES_DCHECK(!gempty_.count(g.first), "non-empty group found in gempty_");
//      MICROSCOPES_DCHECK(counts.at(g.first) == g.second.first, "assignments disagree");
//    }
//  }
//}
