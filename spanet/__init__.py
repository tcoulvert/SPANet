from spanet.network.jet_reconstruction import JetReconstructionModel
from spanet.network.jet_reconstruction_pairwise import JetReconstructionModelWithPairwise
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
    JetReconstructionModel or JetReconstructionModelWithPairwise
        The appropriate model class based on options.use_pairwise_interactions
    """
    if getattr(options, 'use_pairwise_interactions', False):
        return JetReconstructionModelWithPairwise(options, torch_script)
    else:
        return JetReconstructionModel(options, torch_script)
