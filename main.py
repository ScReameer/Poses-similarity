from src.coach import VirtualCoach
coach = VirtualCoach()
def main():
    mode = input('Select mode: video or image: ')
    reference_path = input('Reference path: ')
    actual_path = input('Actual path: ')
    name = input('Output name: ')
    result = coach.compare_poses(reference_path, actual_path, mode, name)
    print('Processing...')
    print(result)
    
if __name__ == '__main__':
    main()