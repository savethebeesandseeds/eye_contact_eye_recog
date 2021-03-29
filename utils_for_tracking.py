import cv2
import sys
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('(major_ver, minor_ver, subminor_ver)', (major_ver, minor_ver, subminor_ver))
if __name__ == '__main__' :
    frame_jump = 60
    # Set up tracker.
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[7]
    # 0 not found
    # 1 works well but slow, never desapears
    # 2 works bad when face changes oriantation
    # 3 not found
    # 4 not found
    # 5 not found
    # 6 not found
    # 7 works very well

    OPENCV_OBJECT_TRACKERS = {
        'csrt': cv2.TrackerCSRT_create,
        'kcf': cv2.TrackerKCF_create,
        # 'boosting': cv2.TrackerBoosting_create,
        'mil': cv2.TrackerMIL_create,
        # 'tld': cv2.TrackerTLD_create,
        # 'medianflow': cv2.TrackerMedianFlow_create,
        # 'mosse': cv2.TrackerMOSSE_create
    }
    # trackers = cv2.legacy.MultiTracker()
    trackers = cv2.MultiTracker()

    # Read video
    # video = cv2.VideoCapture('../DATA/classroom.mp4')
    fn_cnt = 0
    video = cv2.VideoCapture('../DATA/vid_1.mp4')

    # Exit if video not opened.
    if not video.isOpened():
        print('Could not open video')
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    frame = cv2.resize(frame,(0,0),fx = 0.5, fy = 0.5)
    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    for i in range(3):
        # Uncomment the line below to select a different bounding box
        bbox = cv2.selectROI(frame, False)
        print(bbox)
        # Initialize tracker with first frame and bounding box
        tracker = OPENCV_OBJECT_TRACKERS[tracker_type.lower()]()
        trackers.add(tracker, frame, bbox)

    while True:
        # Read a new frame
        video.set(cv2.CAP_PROP_POS_FRAMES, fn_cnt)
        ok, frame = video.read()
        if not ok:
            break
        frame = cv2.resize(frame,(0,0),fx = 0.5, fy = 0.5)
        
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        (success, bboxes) = trackers.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        print("frame: {}, success:{}, bboxes:{}".format(fn_cnt, success, bboxes))
        # Draw bounding box
        if success:
            for bbox in bboxes:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            # Tracking failure
            cv2.putText(frame, 'Tracking failure detected', (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + ' Tracker', (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
    
        # Display FPS on frame
        cv2.putText(frame, 'FPS : ' + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

        # Display result
        cv2.imshow('Tracking', frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
        fn_cnt += frame_jump