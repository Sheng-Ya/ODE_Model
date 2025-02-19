import cProfile
import pstats
import Run_Model


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        # Run your main script
        Run_Model.simulate()   # Ensure this runs the whole model

    print("Finished Run_Model.main()")
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')  # Sort by total time spent in functions
    stats.print_stats(20)
    stats.dump_stats("profiling_results.prof")