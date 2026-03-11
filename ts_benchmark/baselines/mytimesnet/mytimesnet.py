from ts_benchmark.baselines.deep_forecasting_model_base import DeepForecastingModelBase
from ts_benchmark.baselines.mytimesnet.mytimesnet_model import MyTimesNet
from ts_benchmark.baselines.mytimesnet.mytimesnet_model_autovalor import MyTimesNet as MyTimesNetAutovalor
from ts_benchmark.baselines.mytimesnet.mytimesnet_model_alpha_aprendivel import MyTimesNet as MyTimesNetAlpha
from ts_benchmark.baselines.mytimesnet.mytimesnet_model_original import MyTimesNet as MyTimesNetOriginal

# hiperparâmetros padrão
MODEL_HYPER_PARAMS = {
    "enc_in": 1,
    "dec_in": 1,
    "c_out": 1,
    "e_layers": 2,
    "d_layers": 1,
    "d_model": 512,
    "d_ff": 2048,
    "embed": "timeF",
    "freq": "h",
    "dropout": 0.1,
    "batch_size": 32,
    "lr": 0.0001,
    "num_epochs": 10,
    "num_workers": 0,
    "loss": "MSE",
    "patience": 3,
    "task_name": "short_term_forecast",
    "top_k": 5,
    "label_len": 48,
    "pred_len": 24,
    "num_kernels": 6,

}


class MyTimesNetAdapter(DeepForecastingModelBase):

    def __init__(self, **kwargs):
        super().__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "MyTimesNet"

    def _init_model(self):
        return MyTimesNet(self.config)

    def _process(self, input, target, input_mark, target_mark):
        dec_input = target  # simplificado para começar

        output = self.model(
            input,
            input_mark,
            dec_input,
            target_mark
        )

        return {"output": output}

class MyTimesNetOriginalAdapter(DeepForecastingModelBase):

    def __init__(self, **kwargs):
        super(MyTimesNetOriginalAdapter, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "MyTimesNetOriginal"

    def _init_model(self):
        return MyTimesNetOriginal(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input, input_mark, target, target_mark)
        return {"output": output}           


class MyTimesNetAutovalorAdapter(DeepForecastingModelBase):

    def __init__(self, **kwargs):
        super(MyTimesNetAutovalorAdapter, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "MyTimesNetAutovalor"

    def _init_model(self):
        return MyTimesNetAutovalor(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input, input_mark, target, target_mark)
        return {"output": output}                  

class MyTimesNetAlphaAdapter(DeepForecastingModelBase):

    def __init__(self, **kwargs):
        super(MyTimesNetAlphaAdapter, self).__init__(MODEL_HYPER_PARAMS, **kwargs)

    @property
    def model_name(self):
        return "MyTimesNetAlpha"

    def _init_model(self):
        return MyTimesNetAlpha(self.config)

    def _process(self, input, target, input_mark, target_mark):
        output = self.model(input, input_mark, target, target_mark)
        return {"output": output}                          