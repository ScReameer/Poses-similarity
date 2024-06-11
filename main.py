from src.coach import VirtualCoach

COACH = VirtualCoach()

def main():
    mode = input('Select mode: video or image: ')
    reference_path = input('Reference path: ')
    actual_path = input('Actual path: ')
    name = input('Output name: ')
    print('Processing...')
    result = COACH.compare_poses(
        reference_path=reference_path, 
        actual_path=actual_path, 
        mode=mode, 
        name=name
    )
    print(result)
    
if __name__ == '__main__':
    main()