from util.import_util import script_imports

script_imports()

from modules.module.WDModel import WDModel
from modules.module.BlipModel import BlipModel
from modules.module.Blip2Model import Blip2Model
from modules.module.BaseImageCaptionModel import BaseImageCaptionModel
from modules.util.args.GenerateCaptionsArgs import GenerateCaptionsArgs

import torch


def main():
    

    capmodel_list = BaseImageCaptionModel.get_all_model_choices()
    args = GenerateCaptionsArgs.parse_args(capmodel_list.keys())

    if not args.model: # should be redundant, but...
        print("ERROR: No model specified")
        exit(1)

    modelname=args.model
    modeltype=capmodel_list[modelname]

    model = modeltype(torch.device(args.device), args.dtype.torch_dtype(), modelname)

    model.caption_folder(
        sample_dir=args.sample_dir,
        initial_caption=args.initial_caption,
        caption_prefix=args.caption_prefix,
        caption_postfix=args.caption_postfix,
        mode=args.mode,
        error_callback=lambda filename: print("Error while processing image " + filename),
        include_subdirectories=args.include_subdirectories
    )


if __name__ == "__main__":
    main()
