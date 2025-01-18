# from run_models import run_all_models
# from run_new import run


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import torch

    print(torch.cuda.is_available())  # Should return True
    print(torch.cuda.device_count())  # Should show the number of GPUs
    print(torch.version.cuda)  # Should return the CUDA version PyTorch is using

    # run_all_models()
    # run()
    # create_summaries()
