import { resolve, join } from "path";
import { execSync, execFileSync, spawn } from "child_process";

async function main() {
    const parentDir = resolve(__dirname, "..");

    const modelsPath = join(parentDir, "model/build").replaceAll("\\", "/");
    const modelPath = await execSync(
        `ls ${modelsPath}/*.pt | sort -r | head -n 1`
    )
        .toString()
        .trim();

    const exePath = join(parentDir, "cpp/build/Release/simple-torchscript.exe");
    // const data = await execFileSync(exePath, [modelPath]).toString();
    // console.log(data);

    const process = spawn(exePath, [modelPath]);

    process.stdout.on("data", (data) => {
        console.log(data.toString());
    });
}

main();
