"""Synthetic CUDA forward/backward benchmark; no data downloads or training.

python -m experiments.imagenet100.benchmark_multiprobe_look --batch-size 8
"""
import argparse
import gc
import json
import statistics
from pathlib import Path
import torch
from model.deit_tiny_rot_hex_look import DeiTTinyRotHexLook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--steps",type=int,default=8)
    parser.add_argument("--output",type=Path,default=Path("runs/multiprobe_benchmark.json"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    results=[]
    for name,im,fm,rot in (("legacy",1,1,False),("rotating_m1_m1",1,1,True),
                            ("rotating_m1_m4",1,4,True),("rotating_m4_m4",4,4,True),
                            ("independent_m4_m4",4,4,False)):
        torch.manual_seed(0)
        model=DeiTTinyRotHexLook(use_pos_embed=True,directions=6,global_directions=12,
            angular_bins_per_radius=3,look_compact_variable_rings=True,
            center_pose_grid_look=True,center_look_layers_per_probe=3,
            image_look_probes=im,feature_look_probes=fm,feature_look_rotating_probes=rot).cuda()
        # Nonzero fields exercise detector gradients; zero init hides this signal.
        with torch.no_grad():
            model.look_bank.look_grid.normal_(std=.02)
            model.center_look.look_grid.normal_(std=.02)
        x=torch.randn(args.batch_size,3,224,224,device="cuda")
        def step():
            model.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16):
                logits=model(x)
                loss=logits.float().square().mean()
            loss.backward()
            return logits
        for _ in range(3):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(args.steps):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(); logits=step(); end.record(); end.synchronize()
            times.append(start.elapsed_time(end))
        finite=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
        result={"name":name,"parameters":sum(p.numel() for p in model.parameters()),
            "train_step_ms_median":statistics.median(times),"times_ms":times,
            "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
            "finite_gradients":finite,"tokens":model.patch_embed.num_patches}
        # Inference measured separately, excluding allocations retained by backward.
        model.zero_grad(set_to_none=True)
        model.eval()
        forwards=[]
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.float16):
            for _ in range(3): model(x)
            for _ in range(args.steps):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record(); model(x); end.record(); end.synchronize()
                forwards.append(start.elapsed_time(end))
        result["inference_ms_median"]=statistics.median(forwards)
        results.append(result)
        print(json.dumps(result),flush=True)
        del model,x,logits,step
        gc.collect(); torch.cuda.empty_cache()
    output={"gpu":torch.cuda.get_device_name(),"batch_size":args.batch_size,
            "dtype":"float16 AMP", "warmup":3,"results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2),encoding="utf-8")


if __name__=="__main__":main()
