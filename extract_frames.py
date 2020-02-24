import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import argparse
import pickle


#Load keypoints computed by Openpose from every frame into a numpy array
# /!\ Be careful about .DS_STORE into directories /!\
def load_data_from_dir(directory, clean=False, corrected=True):
    data = []
    pose_relative = []
    pose_abs = []
    corresponding_frame = []
    indices = np.arange(75)
    #Remove the confidence score computed by Openpose for every points
    remove_indices = [3*i-1 for i in range(1,26)]
    directory = "./"+str(directory)+"/"
    pose_nb_rel = 0
    pose_nb_abs = 0
    frame_nb = 0
    for filename in sorted(os.listdir(directory)):
        if(filename == ".DS_STORE") or filename.endswith(".mp4"):
            continue
        with open(directory+filename, 'r', encoding = 'utf-8') as f:
            openpose_dict = json.load(f)
            pose_nb_rel = 0
            for vector in openpose_dict['people']:
                keypoint_list = np.array([i for j, i in enumerate(vector['pose_keypoints_2d']) if j not in remove_indices])
                if(clean and -1 in keypoint_list):
                    pose_nb_rel += 1
                    pose_nb_abs += 1
                    continue
                else:
                    if(corrected and -1 in keypoint_list):
                            x_pt = np.array([keypoint_list[2*i] for i in range(25)])
                            y_pt = np.array([keypoint_list[2*i+1] for i in range(25)])
                            x_mean = np.mean(x_pt[x_pt!=-1])
                            y_mean = np.mean(y_pt[y_pt!=-1])
                            x_pt = np.where(x_pt==-1., x_mean, x_pt)
                            y_pt = np.where(y_pt==-1., y_mean, y_pt)
                            keypoint_list = np.stack((x_pt,y_pt)).ravel('F')
                    data.append(keypoint_list)
                    pose_relative.append(pose_nb_rel)
                    pose_abs.append(pose_nb_abs)
                    corresponding_frame.append(frame_nb)
                    pose_nb_rel += 1
                    pose_nb_abs += 1
            frame_nb += 1
    data = np.array(data)
    np.savez_compressed('Xtrain_cluster.npz', data)
    return(data, np.array(pose_relative), np.array(corresponding_frame), np.array(pose_abs))

#Load n array of keypoints computed by Openpose from a given frame into n numpy array
def load_data_from_frame(directory, frame_nb, clean=False, corrected=True):
    data = []
    indices = np.arange(75)
    #Remove the confidence score computed by Openpose for every points
    remove_indices = [3*i-1 for i in range(1,26)]
    base_str = "000000000000"
    file_nb = base_str + str(frame_nb)
    filename ="./"+str(directory) +"/"+str(directory)+"_"+file_nb[-12:]+ "_keypoints.json"
    with open(filename, 'r') as f:
        openpose_dict = json.load(f)
        for vector in openpose_dict['people']:
                keypoint_list = [i for j, i in enumerate(vector['pose_keypoints_2d']) if j not in remove_indices]
                if(clean and -1 in keypoint_list):
                    continue
                else:
                    if(corrected and -1 in keypoint_list):
                        x_pt = np.array([keypoint_list[2*i] for i in range(25)])
                        y_pt = np.array([keypoint_list[2*i+1] for i in range(25)])
                        x_mean = np.mean(x_pt[x_pt!=-1])
                        y_mean = np.mean(y_pt[y_pt!=-1])
                        x_pt = np.where(x_pt==-1., x_mean, x_pt)
                        y_pt = np.where(y_pt==-1., y_mean, y_pt)
                        keypoint_list = np.stack((x_pt,y_pt)).ravel('F')
                    data.append(keypoint_list)
    return(np.array(data))

#Return the true class from every frame into the directory
def get_y_classification(directory, save_title = "ytrain"):
    fps = 25
    timeline = get_timeline(directory)
    directory_path = "./"+str(directory)+"/"
    ytrain = []
    for element in sorted(os.listdir(directory_path)):
        if(element == ".DS_STORE") or element.endswith(".mp4"):
            continue
        if(directory<100):
            frame_nb = element[3:15]
        else:
            frame_nb = element[4:16]
        frame_vid = int(frame_nb) * 1000/fps
        if(timeline[0] <= frame_vid <= timeline[1] or timeline[2] <= frame_vid <= timeline[3] or frame_vid >= timeline[4]):
            ytrain.append(0)
        else:
            ytrain.append(1)
    ytrain = np.array(ytrain)
    np.savez_compressed(save_title+'.npz', ytrain)
    return(ytrain)

#Get the timeline from manual annotations by video id
def get_timeline(id_video):
    objects = []
    with (open("status_timelines.pk", "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile, encoding='latin1'))
            except EOFError:
                break
    for element in objects[0]:
        if element['id'] == id_video:
            return(element['timeline'].to_frame().index)

# Plot the pose detected by Openpose on given frame by cluster
def plot_clustered_pose(directory, frame_nb, corresponding_frame, poses_abs, bow_dim, kmeans, corrected=True):
    poses = load_data_from_frame(directory, frame_nb, corrected=corrected)
    video_path = "./"+str(directory)+"/"+str(directory)+".mp4"
    vidcap = cv2.VideoCapture(video_path)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_nb)
    ret, frame = vidcap.read()

    abs_ = poses_abs[corresponding_frame == frame_nb]
    clusters = kmeans.labels_[abs_]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    labels = ["Cluster "+str(i) for i in range(bow_dim)]
    legend = np.unique(clusters)

    plt.figure(figsize=(20,10))
    my_dict={}
    #Plot the poses with their cluster's color
    for j in range(len(poses)):
        pair_points = np.split(poses[j], 25)
        c = clusters[j]
        for i in range(25):
            my_dict[c] = plt.scatter([(pair_points[i][0]+1)*640], [(pair_points[i][1]+1)*360], c=colors[c], s=20, label=labels[c])
        plt.annotate(str(j), ((pair_points[2][0]+1)*640, (pair_points[2][1]+1)*360), c='r', textcoords="offset points", xytext=(0,5), ha='center')

    #Plot the centroids on image
    my_dict["Centroids"] = 0
    centroids = kmeans.cluster_centers_
    for i in range(len(centroids)):
        pair_points = np.split(centroids[i], 25)
        for j in range(25):
            my_dict["Centroids"] = plt.scatter([(pair_points[j][0]+1)*640], [(pair_points[j][1]+1)*360], c='w', s=15, marker='x', label='Centroids')
            plt.annotate(str(i), ((pair_points[j][0]+1)*640, (pair_points[j][1]+1)*360), c='w', textcoords="offset points", xytext=(0,5), ha='center')


    legend = list(legend)
    legend.append("Centroids")
    plt.axis("off")
    plt.imshow(frame)
    plt.legend(handles=[my_dict[cluster] for cluster in legend], loc='best')
    plt.title("Video "+str(directory)+" Frame "+str(frame_nb))
    frame_title = "000000"+str(frame_nb)
    frame_title = frame_title[len(str(frame_nb)):]
    plt.savefig("./clus_frame/Video"+str(directory)+"_frame"+frame_title+".png", bbox_inches='tight')
    #plt.show()
    plt.close()



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="Video ID to record")
    ap.add_argument("-s", "--start_frame", help="start frame of the record")
    ap.add_argument("-e", "--end_frame", type=int, default=1, help="End frame of the record")
    args = vars(ap.parse_args())

    train_dir = int(args["video"])
    start_frame = int(args["start_frame"])
    end_frame = int(args["end_frame"])

    Xcluster, poses_rel, corresponding_frame, poses_abs = load_data_from_dir(train_dir)
    y_per_frame = get_y_classification(train_dir)
    kmeans = KMeans(n_clusters=10, random_state=0).fit(Xcluster)

    for i in range(start_frame, end_frame):
        plot_clustered_pose(train_dir, i, corresponding_frame, poses_abs, 10, kmeans)
