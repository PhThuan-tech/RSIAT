def get_model(model_name, args):
    name = model_name.lower()
    if name in {"adapter", "umt_adapter"}:
        from models.RSIAT_adapter import Learner
        return Learner(args)
    else:
        assert 0
