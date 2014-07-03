#include <microscopes/mixture/model.hpp>
#include <microscopes/common/dataview.hpp>
#include <microscopes/models/distributions.hpp>
#include <microscopes/common/random_fwd.hpp>

#include <random>
#include <iostream>

using namespace std;
using namespace distributions;
using namespace microscopes::common;
using namespace microscopes::models;
using namespace microscopes::mixture;

template <typename T>
static string
protobuf_to_string(const T &t)
{
  ostringstream out;
  t.SerializeToOstream(&out);
  return out.str();
}

int
main(void)
{
  const size_t D = 28*28;
  rng_t r(5849343);

  vector<shared_ptr<model>> models;
  for (size_t i = 0; i < D; i++)
    models.emplace_back(distributions_factory<BetaBernoulli>().new_instance());

  state s(1000, models);
  state::message_type m_hp;
  m_hp.set_alpha(2.0);
  s.set_hp(protobuf_to_string(m_hp));

  distributions_model<BetaBernoulli>::message_type m_feature_hp;
  m_feature_hp.set_alpha(1.0);
  m_feature_hp.set_beta(1.0);

  for (size_t i = 0; i < D; i++)
    s.set_feature_hp(i, protobuf_to_string(m_feature_hp));

  const size_t G = 50;

  for (size_t i = 0; i < G; i++)
    s.create_group(r);

  // create fake data
  bool data[D];
  for (size_t i = 0; i < D; i++)
    data[i] = bernoulli_distribution(0.5)(r);

  vector<runtime_type_info> types(D, TYPE_INFO_B);
  vector<size_t> offsets;
  for (size_t i = 0; i < D; i++)
    offsets.push_back(i);

  row_accessor acc( reinterpret_cast<const uint8_t *>(&data[0]), nullptr, &types, &offsets );

  s.add_value(G/2, 10, acc, r);

  float sum = 0.0;
  const size_t NTRIALS = 60000;
  for (size_t i = 0; i < NTRIALS; i++) {
    s.remove_value(10, acc, r);
    const auto p = s.score_value(acc, r);
    sum += p.first[1];
    sum += p.second[0];
    s.add_value(G/2, 10, acc, r);
  }

  cout << "meaningless: " << sum << endl;
  return 0;
}
