import qai_hub as hub

print("1. Load compiled model \n 2. Load TorchScript model")
choice = input("Enter your choice: ")
if choice == '1':
    # Profile the previously compiled model
    profile_job = hub.submit_profile_job(
    model=hub.get_job("jgz2rw0xg").get_target_model(),
    device=hub.Device("QCS8550 (Proxy)"),
    )
    assert isinstance(profile_job, hub.ProfileJob)    

elif choice == '2':
    # Compile previously saved torchscript model
    compile_job = hub.submit_compile_job(
        model=r"pretrained_models\base\scripted_model_ots_4073_9968.pt",
        device=hub.Device("QCS8550 (Proxy)"),
        input_specs=dict(image=(1, 3, 224, 224)),
    )
    assert isinstance(compile_job, hub.CompileJob)

    profile_job = hub.submit_profile_job(
        model=compile_job.get_target_model(),
        device=hub.Device("QCS8550 (Proxy)"),
    )
    assert isinstance(profile_job, hub.ProfileJob)