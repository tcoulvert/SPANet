from spanet.network.jet_reconstruction import JetReconstructionModel
from spanet.dataset import JetReconstructionDataset
from spanet.options import Options
from spanet.interface import SPANetInterface


def create_model(options: Options, torch_script: bool = False):
    """Factory function to create the appropriate model based on options.

    Parameters
    ----------
    options : Options
        SPANet configuration options
    torch_script : bool
        Whether to compile with TorchScript

    Returns
    -------
    JetReconstructionModel
        Jet reconstruction model. If options.use_pairwise_interactions is True,
        the pairwise-aware hooks inside the jet reconstruction network are used.
    """
    return JetReconstructionModel(options, torch_script)
