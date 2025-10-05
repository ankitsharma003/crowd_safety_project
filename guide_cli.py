import argparse
import json
from flow_estimation import load_images, estimate_flow
from bottleneck_detection import detect_bottlenecks
from movement_instruction import generate_instructions, generate_instruction_details


def main():
    parser = argparse.ArgumentParser(description="Crowd flow guidance CLI")
    parser.add_argument("folder", help="Folder with frames (e.g., UCSDped1/Test/Test001)")
    parser.add_argument("--max_frames", type=int, default=20, help="Max frames to load")
    parser.add_argument("--alpha", type=float, default=0.75, help="Smoothing factor (0-1)")
    parser.add_argument("--density_threshold", type=float, default=0.55, help="Bottleneck density threshold (0-1)")
    parser.add_argument("--movement_threshold", type=float, default=0.04, help="Bottleneck movement threshold (0-1)")
    args = parser.parse_args()

    images = load_images(args.folder, max_frames=args.max_frames)
    flows = estimate_flow(images)
    bottlenecks = detect_bottlenecks(images, flows, density_threshold=args.density_threshold, movement_threshold=args.movement_threshold)
    instructions = generate_instructions(images, flows, bottlenecks)
    details = generate_instruction_details(images, flows, bottlenecks, alpha=args.alpha)

    output = {
        "folder": args.folder,
        "bottlenecks": bottlenecks,
        "instructions": instructions,
        "details": details,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()


