""" Evaluate best model given a trained model. """
import logging
import glob
from .pipeline import VTACMLPipe
from .utils import ROOTDIR

if __name__ == "__main__":
    sequence = "output/models/seq_0_1_2_3/"

    sequence_dir = ROOTDIR / sequence

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=f"{sequence_dir}/evaluation_log.txt",
                        filemode='w'
                        )
    log = logging.getLogger(__name__)
    log.info(f"Evaluating {sequence} models:")

    for path in glob.glob(f"{sequence_dir}/*.pkl") :
        seq, model = path.split('/')[-2:]
        # print(seq, model)
        pipe = VTACMLPipe(config_file=f'config/config_{seq}.yaml')
        pipe.load_model(model_name=model)

        pipe.evaluate(name=f"{seq}/{model}", plot=False)
