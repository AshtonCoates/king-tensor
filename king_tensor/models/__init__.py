def create_model(version: str, num_classes: int):

    if version == "v1":
        from .v1 import SmallCardNet, transform, model_path
        return SmallCardNet(num_classes), transform, model_path

    else:
        raise ValueError(f"Unknown model version: {version}")
