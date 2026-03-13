def cal_model_size(model):
    num_bytes = 0
    num_param = 0
    for p in model.parameters():
        num_param += p.numel()
        num_bytes += p.numel() * p.element_size()
    print("params:", num_param/1e9, "B")
    print("bytes:", num_bytes/1024**3, "GB")
    return (num_param,'个'), (num_bytes,'bytes')

