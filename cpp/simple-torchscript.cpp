#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char *argv[])
{
	if (argc != 2)
	{
		std::cerr << "usage: example-app <path-to-exported-script-module>\n";
		return -1;
	}

	torch::jit::script::Module module;
	try
	{
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load(argv[1]);

		// Create a vector of inputs.
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(torch::rand({1, 10}));

		auto output1 = module.get_method("forward_fc1")(inputs);
		auto output2 = module.get_method("forward_fc2")({output1});
		auto output3 = module.get_method("forward_fc3")({output2});

		// Execute the model and turn its output into a tensor.
		at::Tensor output = module.forward(inputs).toTensor();

		std::cout << output1 << std::endl;
		std::cout << output2 << std::endl;
		std::cout << output3 << std::endl;
		std::cout << output << std::endl;
	}
	catch (const c10::Error &e)
	{
		std::cerr << "error loading the model\n";
		return -1;
	}

	std::cout << "ok\n";
}