import hydra


def get_current_hydra_output_dir():
    conf = hydra.core.hydra_config.HydraConfig.get()
    return conf.runtime.output_dir
