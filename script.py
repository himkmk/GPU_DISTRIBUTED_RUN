import os
import time
from multiprocessing import Process
from itertools import product



def generate_script(CUDA, opt):

    # save dir of NST
    # /mnt/ssd0/mklee/NeuralStyleTransfer/data/results/2022_05_05/@INPUTMODEL/@DATASET/@OPTIMIZER_@LR_@INPUTS


    _cuda = f"CUDA_VISIBLE_DEVICES={CUDA}"
    _python = "/opt/conda/bin/python3"
    _runfile = "test.py"
    _option = [
        "--blend_type", opt["blend_type"],
        "--testset", opt["testset"],
        "--save_folder", "test_results",
        "--resume", "./pretrained_weights/masa.pth",  # masa.pth|masa_rec.pth
        "--name", "percep",  # percep|psnr respect to model type (arg.resume)
        "--ref_type", "SST",
        "--test_config", ":".join([opt["test_name"], opt["input_model"], opt["dataset"],
                                  opt["optimizer"]+"_"+opt["lr"]+"lr"+"_"+opt["style_img"]+opt["init_img"],
                                  opt["postfix"]])
    ]

    script = " ".join([_cuda, _python, _runfile, *_option])
    print("FOUND GPU--------------", CUDA)
    print(script.replace("--save_dir ", ""))

    return script

CUDA_FREE_FLAGS = {
    "0": {"free":True, "process": None},
    "1": {"free":True, "process": None},
    "2": {"free":True, "process": None},
    "3": {"free":True, "process": None},
}


# ignore order, order define below in variable "order"
config_pool = {
    "blend_type": ["percep"],
    "testset": ["SST_DATASET"],
    "postfix": ["_blend"],
    "test_name": ["2022_05_24_suppli"],
    "style_img": ["percepx2"],
    "init_img": ["psnrx4"],
    "input_model": ["ESRGAN"],
    "optimizer": ["500Adam"],
    "lr": ["0.01"],
    "dataset": [
        # "set5",
        "DIV2K_val",
        # "set14",
        "bsd100",
        "sun80",
        # "urban100",
        # "celebA",
        # "AFHQ"
    ],
}

order = [
    "dataset",
    "blend_type",
    "testset",
    "test_name",
    "postfix",
    "style_img",
    "init_img",

    "input_model",
    "optimizer",
    "lr",
]


def GPU_DISTRIBUTED_RUN(config_pool, order):
    assert sorted(order)==sorted(config_pool.keys())
    for configs in list(product(*[config_pool[key] for key in order])):

        opt = {}
        for key_, config_, in zip(order, configs):
            opt[key_] = config_


        processes = []
        while True:  # while finding free GPU to assign process

            BREAK_FLAG = False

            # check gpus to assign new process
            for gpu_id in CUDA_FREE_FLAGS:

                # if gpu is free, assing new process and set CUDA_FREE_FLAG = False

                if CUDA_FREE_FLAGS[gpu_id]["free"]:
                    script = generate_script(gpu_id, opt)
                    p = Process(target=os.system, args=(script,))
                    p.start()
                    CUDA_FREE_FLAGS[gpu_id] = {"free":False, "process":p}
                    BREAK_FLAG = True
                    break

            # check gpus for finished process
            for gpu_id in CUDA_FREE_FLAGS:
                if not CUDA_FREE_FLAGS[gpu_id]["free"]:
                    is_alive = CUDA_FREE_FLAGS[gpu_id]["process"].is_alive()

                    # if process finished, free gpu and kill process
                    if not is_alive:
                        CUDA_FREE_FLAGS[gpu_id]["process"].join()
                        CUDA_FREE_FLAGS[gpu_id]["free"] = True

            if BREAK_FLAG:
                BREAK_FLAG = False
                break

            else:
                time.sleep(1)

    for gpu_id in CUDA_FREE_FLAGS:
        CUDA_FREE_FLAGS[gpu_id]["process"].join()
        p.join()

    print("\n\n\n\nEND?\n\n\n\n")



GPU_DISTRIBUTED_RUN(config_pool, order)
GPU_DISTRIBUTED_RUN(config_pool, order)
