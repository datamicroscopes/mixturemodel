#include <microscopes/mixture/model.hpp>
#include <microscopes/common/recarray/dataview.hpp>
#include <microscopes/models/distributions.hpp>
#include <microscopes/common/random_fwd.hpp>

#include <random>
#include <iostream>

using namespace std;
using namespace distributions;
using namespace microscopes;
using namespace microscopes::common;
using namespace microscopes::common::recarray;

int
main(void)
{
  const size_t D = 28*28;
  rng_t r(5849343);

  vector<shared_ptr<models::model>> models;
  for (size_t i = 0; i < D; i++)
    models.emplace_back(make_shared<
        models::distributions_model<BetaBernoulli>>());

  mixture::model_definition def(models);

  shared_ptr<mixture::state> spx(
      mixture::state::unsafe_initialize(def, 1000));
  auto s = *spx;
  s.get_cluster_hp_mutator("alpha").set<float>(2.0);

  for (size_t i = 0; i < D; i++) {
    s.get_feature_hp_mutator(i, "alpha").set<float>(1.0);
    s.get_feature_hp_mutator(i, "beta").set<float>(1.0);
  }

  //const size_t G = strtoul(argv[1], nullptr, 10);
  //cout << "groups: " << G << endl;
  const size_t G = 5;

  for (size_t i = 0; i < G; i++)
    s.create_group(r);

  // create fake data
  bool data[D];
  for (size_t i = 0; i < D; i++)
    data[i] = bernoulli_distribution(0.5)(r);

  vector<runtime_type> types(D, runtime_type(TYPE_B));

  row_accessor acc( reinterpret_cast<const uint8_t *>(&data[0]), nullptr, &types);

  s.add_value(G/2, 10, acc, r);

  float sum = 0.0;
  const size_t NTRIALS = 100;
  for (size_t i = 0; i < NTRIALS; i++) {
    const size_t gid = s.remove_value(10, acc, r);
    s.delete_group(gid);
    s.create_group(r);
    const auto p = s.score_value(acc, r);
    sum += p.first[1];
    sum += p.second[0];
    const auto groups = s.groups();
    const vector<float> probs(groups.size(), 1./float(groups.size()));
    const auto choice = util::sample_discrete(probs, r);
    s.add_value(groups[choice], 10, acc, r);
  }
  s.dcheck_consistency();

  //cout << "meaningless: " << sum << endl;
  return 0;
}
