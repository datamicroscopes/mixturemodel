from microscopes.mixture.model import sample
from microscopes.mixture.definition import model_definition
from microscopes.models import bb, nich

from distributions.dbg.models import nich as dbg_nich


def test_sample_sanity():
    # just a sanity check
    defn = model_definition(10, [bb, nich])
    sample(defn)
    sample(defn,
           cluster_hp={'alpha': 10.},
           feature_hps=[
               {'alpha': 54.3, 'beta': 34.5},
               dbg_nich.EXAMPLES[0]['shared']])
