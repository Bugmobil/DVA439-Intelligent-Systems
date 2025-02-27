import qai_hub as hub

def profile_model():
    if input("Load from\n1. QAI model \n2. Local model\n?: ") == '1':
        # Load the model from QAI model
        qai_model = hub.get_job(input("Enter job code: ")).get_target_model()
        # Profile the previously compiled model
        profile_job = hub.submit_profile_job(
        model=qai_model,
        device=hub.Device("QCS8550 (Proxy)"),
        )
        assert isinstance(profile_job, hub.ProfileJob)    

    else:
        model_path = input("Enter the path to the model: ")

        # Compile previously saved torchscript model
        compile_job = hub.submit_compile_job(
        model=model_path,
        device=hub.Device("QCS8550 (Proxy)"),
        input_specs=dict(image=(1, 3, 256, 256)),
        )
        assert isinstance(compile_job, hub.CompileJob)
    
        profile_job = hub.submit_profile_job(
            model=compile_job.get_target_model(),
            device=hub.Device("QCS8550 (Proxy)"),
        )
        assert isinstance(profile_job, hub.ProfileJob)