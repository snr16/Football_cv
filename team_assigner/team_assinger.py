from sklearn.cluster import KMeans
import numpy as np

class TeamAssinger():

    def __init__(self):
        self.kmeans = None
        self.team_colors={}
        self.player_team_dict = {}

    def get_clustering_model(self,image):
        #reshape the image
        image_2d = image.reshape(-1,3)
        kmeans = KMeans(n_clusters=2,init='k-means++',n_init=1,random_state=42)
        kmeans.fit(image_2d)

        return kmeans
    
    def get_player_color(self,frame,bbox):
        image = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        #crop the top half of image (HxW)
        top_half_img = image[0:int(image.shape[0]/2),:,:]

        kmeans = self.get_clustering_model(top_half_img)

        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_img.shape[0],top_half_img.shape[1])

        corners = [clustered_image[0,0],
            clustered_image[0,-1],
            clustered_image[-1,0],
            clustered_image[-1,-1]]

        non_player_cluster = np.max(corners)
        player_cluster = 1-non_player_cluster


        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self,frame,player_detections):
        player_colors = []
        for _,player in player_detections.items():
            player_color = self.get_player_color(frame,player['bbox'])
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2,init='k-means++',n_init=1,random_state=42)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1]=kmeans.cluster_centers_[0]
        self.team_colors[2]=kmeans.cluster_centers_[1]

    def assign_player_team(self,frame,player_id,bbox):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame,bbox)
        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0] + 1
        self.player_team_dict[player_id]=team_id

        return team_id

