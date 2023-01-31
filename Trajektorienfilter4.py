#datenaufbreitung: merging der zerteilten trajektorienzu Trajektorie mit 1799 frames an daten
import csv
import matplotlib.pyplot as plt
import numpy as np


def store_data(complete_data):
    all_id_data = []
    np_dict = dict()
    for every_id in range(len(complete_data)):
        id_data = {"Walking":np.array(complete_data[every_id],dtype=np.float32)}
        all_id_data.append(id_data)
    for i in range(len(all_id_data)):
        np_dict['S'+str(i+1)] = all_id_data[i]
    np_dict2 = {'positions_3D':np_dict}
    np.savez("data_3d_h36m.npz",positions_3D = np_dict2)

    import sys
    data = np.load("data_3d_h36m.npz",allow_pickle = True)
    print(data.files)
    row = data.files
    np.set_printoptions(threshold=np.inf)
    #print(data['arr_0'])
    sys.stdout=open("data_3d_h36m.txt","w")
    for i in row:
        print(data[i])
    sys.stdout.close()           


def calc_number_of_skeletons(sequence_number):
    g = 0
    with open('D:/Studium/WS22/Konfigurationen Datensets/Usable sequences/BestBatch/seq_' + str(sequence_number) +'/coords.csv') as dataFile:
        heading = next(dataFile)
        readData = csv.reader(dataFile)
        for row in readData:
            g += 1
    #skeleton_over_all_frames = np.zeros((15811,32,3))
    #skeleton_over_all_frames = np.zeros((g,32,3))
    return g

def calc_skeleton_sequences(sequence_number,ROI_limitations):
    (x_max,x_min,y_max,y_min) = ROI_limitations
    joints_coords_within_frame = np.zeros((32,3))
    personIDsInit = []
    personPosXinit = []
    personPosYinit = []
    allPeople = []
    complete_data = []
    n = []
    i = 0
    z = 0

    with open('D:/Studium/WS22/Konfigurationen Datensets/Usable sequences/BestBatch/seq_' + str(sequence_number) + '/coords.csv') as dataFile:
        heading = next(dataFile)
        readData = csv.reader(dataFile)
        for row in readData: 
            point_position = []
            if row[2] == '0' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[14][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360) 
                joints_coords_within_frame[14][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[14][2] = (float(row[7]) - float(row[12]))     
            elif row[2] == '1' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[13][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[13][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[13][2] = (float(row[7]) - float(row[12]))      
            elif row[2] == '2' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[12][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[12][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[12][2] = (float(row[7]) - float(row[12]))
                '''
            elif row[2] == '3' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                point_position.append((float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360))
                point_position.append((float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360))
                point_position.append((float(row[7]) - float(row[12])))
                #joints_coords_within_frame[14] = point_position   
            '''     
            elif row[2] == '4' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[24][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[24][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[24][2] = (float(row[7]) - float(row[12]))     
            elif row[2] == '5' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[25][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[25][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[25][2] = (float(row[7]) - float(row[12]))
            elif row[2] == '6' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[26][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[26][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[26][2] = (float(row[7]) - float(row[12]))      
                '''
            elif row[2] == '7' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                point_position.append((float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360))
                point_position.append((float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360))
                point_position.append((float(row[7]) - float(row[12]))) 
                #joints_coords_within_frame[14] = point_position 
            '''     
            elif row[2] == '8' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[16][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[16][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[16][2] = (float(row[7]) - float(row[12])) 
            elif row[2] == '9' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[17][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[17][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[17][2] = (float(row[7]) - float(row[12]))    
            elif row[2] == '10' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[18][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[18][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[18][2] = (float(row[7]) - float(row[12]))       
            elif row[2] == '11' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[11][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[11][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[11][2] = (float(row[7]) - float(row[12])) 
                '''
            elif row[2] == '12' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                point_position.append((float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360))
                point_position.append((float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360))
                point_position.append((float(row[7]) - float(row[12])))    
                #joints_coords_within_frame[11] = point_position 
            elif row[2] == '13' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                point_position.append((float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360))
                point_position.append((float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360))
                point_position.append((float(row[7]) - float(row[12]))) 
                #joints_coords_within_frame[11] = point_position      
            elif row[2] == '14' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                point_position.append((float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360))
                point_position.append((float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360))
                point_position.append((float(row[7]) - float(row[12])))        
                #joints_coords_within_frame[11] = point_position
            '''  
            elif row[2] == '15' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[0][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[0][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[0][2] = (float(row[7]) - float(row[12]))     
            elif row[2] == '16' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[1][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[1][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[1][2] = (float(row[7]) - float(row[12]))      
            elif row[2] == '17' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[2][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[2][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[2][2] = (float(row[7]) - float(row[12]))    
            elif row[2] == '18' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[5][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[5][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[5][2] = (float(row[7]) - float(row[12])) 
            elif row[2] == '19' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[6][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[6][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[6][2] = (float(row[7]) - float(row[12]))  
            elif row[2] == '20' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[7][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[7][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[7][2] = (float(row[7]) - float(row[12]))   
            elif row[2] == '21' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                joints_coords_within_frame[10][0] = (float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[10][1] = (float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360)
                joints_coords_within_frame[10][2] = (float(row[7]) - float(row[12]))  

            if row[2] == '21' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                #skeleton_over_all_frames.append(joints_coords_within_frame)
                skeleton_over_all_frames[z] = joints_coords_within_frame
                z += 1
            if row[2] == '12' and len(n) == (int(row[0])-1) and int(row[0]) > 0:
                personPosXinit.append((float(row[5]) - float(row[10]))*np.cos(2*np.pi*float(row[15])/360))
                personPosYinit.append((float(row[6]) - float(row[11]))*np.sin(2*np.pi*float(row[15])/360))         
                personIDsInit.append(row[1])
                i += 1
            if len(n) < (int(row[0]) - 1):
                n.append(i)
                i = 0

        personPosX = []
        personPosY = []
        personIDs  = []
        skeletons_within_a_frame = []
        r = 0
        for k in range(len(n)):
            personPosX.append(personPosXinit[r:r+n[k]])
            personPosY.append(personPosYinit[r:r+n[k]])
            personIDs.append(personIDsInit[r:r+n[k]])
            
            temp_skeleton_in_frame = []
            for b in range(n[k]):
                oe = r+b
                temp_skeleton_in_frame.append(skeleton_over_all_frames[oe])
                #print(skeleton_over_all_frames[oe])
            skeletons_within_a_frame.append(temp_skeleton_in_frame)
            #print(skeletons_within_a_frame[k])
            r += n[k]

        for k in range(len(personIDs)):
            for j in range(len(personIDs[k])):
                if personIDs[k][j] not in allPeople:
                    allPeople.append(personIDs[k][j])

        id_position_pairs = []
        allPositions = []
        allSkeletons = []
        jump_in_arrays = []
        NUM_COLORS = 20
        cm = plt.get_cmap('tab20')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

        for l in range(len(allPeople)):
            sortedPositionX = []
            sortedPositionY = []
            sorted_skeletons_within_a_frame = []
            frame_numbers = []
            for f in range(len(personPosX)):
                if l < len(personPosX[f]):
                    if personPosY[f][l] < y_max and personPosY[f][l] > y_min and personPosX[f][l] > x_min and personPosX[f][l] < x_max:
                        sortedPositionX.append(personPosX[f][l])
                        sortedPositionY.append(personPosY[f][l])
                        sorted_skeletons_within_a_frame.append(skeletons_within_a_frame[f][l])          #numFrames x 32 x 3
                        frame_numbers.append(f)

            if len(frame_numbers) != 0:        
                plt.scatter(sortedPositionX,sortedPositionY,label=allPeople[l])
                sortedFrameNumbers = []
                jump_in_array_index = []
                for k in range(len(frame_numbers)):
                    if frame_numbers[k] != frame_numbers[k-1]+1:
                        sortedFrameNumbers.append(frame_numbers[k])
                        sortedFrameNumbers.append(frame_numbers[k-1])
                        jump_in_array_index.append(k-1)
                        jump_in_array_index.append(k)
                sortedFrameNumbers = np.sort(sortedFrameNumbers)
                first_element = jump_in_array_index.pop(0)
                jump_in_array_index.append(first_element)
                print("Person ID:",allPeople[l],"\t","Frames:",sortedFrameNumbers)
                print("Jump in array location:",jump_in_array_index)

                pairs = []
                positions = []
                pairs.append(allPeople[l])
                pairs.append(sortedFrameNumbers)
                id_position_pairs.append(pairs)
                positions.append(sortedPositionX)
                positions.append(sortedPositionY)
                allPositions.append(positions)
                allSkeletons.append(sorted_skeletons_within_a_frame) #numPeople x numFramesOfPerson x 32 x 3
                jump_in_arrays.append(jump_in_array_index)           #numPeople x 2 x numFramesOfPerson 
        
        #for b in range(len(allSkeletons)):
            #print("allSkeletons:",len(allSkeletons),len(allSkeletons[b]),len(allSkeletons[b][0]))
            #print("allPositions:",len(allPositions),len(allPositions[b]),len(allPositions[b][0]))
        #    print(b, " ", allSkeletons[b][-1])
        #    print(b, " ", allPositions[b][0][-1], allPositions[b][1][-1])
        
        plt.legend()
        plt.savefig('D:/Studium/WS22/Konfigurationen Datensets/Used Seq Graphs/Sequence'+str(sequence_number)+'.jpg')
        #plt.show()    

        print(" ")

        set_of_available_nodes = []

        #nodes describe the properties of a coordinate information snippet
        class node:
            def __init__(self,jump_location_start,jump_location_end,x_coord_list,y_coord_list,skeleton_coord_list,node_id,list_position):
                self.start_jump = jump_location_start
                self.end_jump = jump_location_end
                self.x_coordiantes = x_coord_list
                self.y_coordinates = y_coord_list

                self.skeleton_coordinates = skeleton_coord_list

                self.intervall_id = node_id
                self.children_names = []
                self.list_key = list_position
                self.children = []

            def return_intervall_id(self):
                return self.intervall_id
            
            def return_intervall(self):
                return self.start_jump, self.end_jump

            def return_list_key(self):
                return self.list_key
            
            def return_coordinates(self):
                return self.x_coordiantes,self.y_coordinates,self.skeleton_coordinates

            def look_for_children(self):
                q = 0
                for child_node in set_of_available_nodes:
                    child_intervall = child_node.return_intervall()
                    if child_intervall[0] == self.end_jump + 1:
                        self.children_names.append(child_node.return_intervall_id()) #namen sind nicht korrekt wegen index k missing? oder doch korrekt?
                        #self.children.append(child_node)
                        self.children.append(q)
                        #hier die heuristik einfügen
                    q += 1
                #return self.children_names #lieber die children node ausgeben als den namen nur
                return self.children

        #create set of available nodes to connect to the seed routes
        for l in range(len(id_position_pairs)):
            if id_position_pairs[l][1][0] != 0:
                if len(id_position_pairs[l][1])>2:
                    for k in range(0,len(id_position_pairs[l][1]),2):
                        list_index_key = len(set_of_available_nodes)
                        set_of_available_nodes.append(node(id_position_pairs[l][1][k],
                                                        id_position_pairs[l][1][k+1],
                                                        allPositions[l][0][jump_in_arrays[l][k]:jump_in_arrays[l][k+1]],
                                                        allPositions[l][1][jump_in_arrays[l][k]:jump_in_arrays[l][k+1]],
                                                        allSkeletons[l][jump_in_arrays[l][k]:jump_in_arrays[l][k+1]],
                                                        id_position_pairs[l][0] + " " + str(k),
                                                        list_index_key))
                        print("Jumps Node:",k,k+1,jump_in_arrays[l][k],jump_in_arrays[l][k+1],id_position_pairs[l][0] + " " + str(k))
                else:
                    list_index_key = len(set_of_available_nodes)
                    set_of_available_nodes.append(node(id_position_pairs[l][1][0],
                                                    id_position_pairs[l][1][1],
                                                    allPositions[l][0][jump_in_arrays[l][0]:jump_in_arrays[l][1]],
                                                    allPositions[l][1][jump_in_arrays[l][0]:jump_in_arrays[l][1]],
                                                    allSkeletons[l][jump_in_arrays[l][0]:jump_in_arrays[l][1]],
                                                    id_position_pairs[l][0] + " " + str(1),
                                                    list_index_key))
                    print("Jumps Node:",0,1,jump_in_arrays[l][0],jump_in_arrays[l][1],id_position_pairs[l][0] + " " + str(1))
        print(" ")            

        def shortest_path(start,end):                                                                                                                       #complete search algorithm
            queue = [[start]]
            visited = set()
            while queue:
                path = queue.pop(0)                                                                                                                         # Get the path with the highest priority (i.e., the shortest path)
                node = path[-1]                                                                                                                             # Get the last node in the path
                if node not in visited:                                                                                                                     # If the node has not been visited yet
                    current_node = set_of_available_nodes[node.return_list_key()]#hier die namen ausgeben lassen
                    neighbours = current_node.look_for_children()
                    print("Node:",current_node.return_intervall_id(),current_node,neighbours,current_node.return_intervall_id()," | ",current_node.return_list_key()," | ", current_node.return_intervall())
                    for neighbour in neighbours:
                        new_path = list(path)                                                                                                               # Create a new path with the neighbour appended to the current path
                        #new_path.append(neighbour)
                        new_path.append(set_of_available_nodes[neighbour])
                        #ending_check = neighbour.return_intervall()
                        ending_check = set_of_available_nodes[neighbour]
                        #if neighbour.return_intervall_id() == end.return_intervall_id() or ending_check[1] == 1798:                                         # If the neighbour is the end node, return the path
                        last_element_check = ending_check.return_intervall()
                        #if ending_check.return_intervall_id() == end.return_intervall_id() or last_element_check[1] == 1798:
                        if ending_check.return_intervall_id() == end.return_intervall_id() or last_element_check[1] == 1798:    
                            return new_path 
                        queue.append(new_path)                                                                                                              # Else, add the new path to the priority queue
                    visited.add(node)
                    queue = sorted(queue, key=len)
            return None                                                                                                                                     # If no path is found, return None


        from mpl_toolkits import mplot3d
        for l in range(len(id_position_pairs)):
            if id_position_pairs[l][1][0] == 0:                                                                                                             #only take routes that begin with the frame 0        
                if len(id_position_pairs[l][1])>2:
                    #for k in range(1,len(id_position_pairs[l][1])-1,2):                                                                                     #for every seed route, go through all gaps. there, look for best combination of nodes
                    #for k in range(0,len(id_position_pairs[l][1]),2):
                    #
                    total_appended_length = 0
                    complete_trajectory_x = []
                    complete_trajectory_y = []
                    complete_skeleton_trajectory = []
                    #
                    for k in range(0,len(id_position_pairs[l][1])-2,2):
                        list_index_key = len(set_of_available_nodes) 
                        start_node = node(id_position_pairs[l][1][k],
                                        id_position_pairs[l][1][k+1],
                                        allPositions[l][0][jump_in_arrays[l][k]:jump_in_arrays[l][k+1]],
                                        allPositions[l][1][jump_in_arrays[l][k]:jump_in_arrays[l][k+1]],
                                        allSkeletons[l][jump_in_arrays[l][k]:jump_in_arrays[l][k+1]],
                                        id_position_pairs[l][0] + " " + str(k),
                                        list_index_key)
                        set_of_available_nodes.append(start_node)
                        list_index_key = len(set_of_available_nodes)
                        #end_node   = node(id_position_pairs[l][1][k],id_position_pairs[l][1][k+1],allPositions[l][0][jump_in_arrays[l][k]:jump_in_arrays[l][k+1]],allPositions[l][1][jump_in_arrays[l][k]:jump_in_arrays[l][k+1]],id_position_pairs[l][0] + " " + str(k),list_index_key)
                        end_node   = node(id_position_pairs[l][1][k+2],
                                        id_position_pairs[l][1][k+3],
                                        allPositions[l][0][jump_in_arrays[l][k+2]:jump_in_arrays[l][k+3]],
                                        allPositions[l][1][jump_in_arrays[l][k+2]:jump_in_arrays[l][k+3]],
                                        allSkeletons[l][jump_in_arrays[l][k+2]:jump_in_arrays[l][k+3]],
                                        id_position_pairs[l][0] + " " + str(k),
                                        list_index_key)
                        set_of_available_nodes.append(end_node)
                        path = shortest_path(start_node,end_node)
                        
                        print("Output:",id_position_pairs[l][0] + " " + str(k),path)
                        print("Jumps Start:",k,k+1, jump_in_arrays[l][k],jump_in_arrays[l][k+1],id_position_pairs[l][0] + " " + str(k))
                        print("Jumps End:",k+2,k+3, jump_in_arrays[l][k+2],jump_in_arrays[l][k+3],id_position_pairs[l][0] + " " + str(k))
                        print(" ")

                        set_of_available_nodes.remove(start_node)
                        set_of_available_nodes.remove(end_node)

                        #arrays concatenaten & jump locations neu setzen & in scatter graph plotten
                        gap_filler_x = []
                        gap_filler_y = []
                        gap_filler_skeleton = []
                        
                        ###########################################################################
                        if path != None:
                            for x in range(len(path)):
                                gap_node = path[x]
                                coordinates = gap_node.return_coordinates()
                                gap_filler_x += coordinates[0]
                                gap_filler_y += coordinates[1]
                                gap_filler_skeleton += coordinates[2]
                        ##########################################################################

                        total_appended_length += len(gap_filler_x)
                        complete_trajectory_x = allPositions[l][0][(jump_in_arrays[l][k]+total_appended_length) : (jump_in_arrays[l][k+1]+total_appended_length)] + gap_filler_x + allPositions[l][0][(jump_in_arrays[l][k+2]+total_appended_length) : (jump_in_arrays[l][k+3]+total_appended_length)]
                        complete_trajectory_y = allPositions[l][1][(jump_in_arrays[l][k]+total_appended_length) : (jump_in_arrays[l][k+1]+total_appended_length)] + gap_filler_y + allPositions[l][1][(jump_in_arrays[l][k+2]+total_appended_length) : (jump_in_arrays[l][k+3]+total_appended_length)]
                        

                        #das hier umändern die concatenation funktioniert nicht wie gewollt
                        #print(" ")
                        print(len(complete_skeleton_trajectory),len(gap_filler_skeleton))
                        #print(allSkeletons[l][jump_in_arrays[l][k]+total_appended_length : jump_in_arrays[l][k+1]+total_appended_length])
                        #print(gap_filler_skeleton)
                        complete_skeleton_trajectory = allSkeletons[l][(jump_in_arrays[l][k]+total_appended_length) : (jump_in_arrays[l][k+1]+total_appended_length)] + gap_filler_skeleton + allSkeletons[l][(jump_in_arrays[l][k+2]+total_appended_length) : (jump_in_arrays[l][k+3]+total_appended_length)]

                    complete_data.append(complete_skeleton_trajectory)
                    print(len(complete_skeleton_trajectory),len(complete_skeleton_trajectory[0]),len(complete_skeleton_trajectory[0][0]))
                    #plt.scatter(complete_trajectory_x,complete_trajectory_y,label=id_position_pairs[l][0]) 
                    
                    complete_skeleton_trajectory_x = []
                    complete_skeleton_trajectory_y = []
                    complete_skeleton_trajectory_z = []
                    for h in range(len(complete_skeleton_trajectory)):
                        for g in range(32):
                            if complete_skeleton_trajectory[h][g][0] != 0 and complete_skeleton_trajectory[h][g][1] != 0 and complete_skeleton_trajectory[h][g][2] != 0:
                                complete_skeleton_trajectory_x.append(complete_skeleton_trajectory[h][g][0])
                                complete_skeleton_trajectory_y.append(complete_skeleton_trajectory[h][g][1])
                                complete_skeleton_trajectory_z.append(complete_skeleton_trajectory[h][g][2])

                    #plt.scatter(complete_skeleton_trajectory[:][:][0],complete_skeleton_trajectory[:][:][1],complete_skeleton_trajectory[:][:][2],label=id_position_pairs[l][0])
                    #plt.scatter(complete_skeleton_trajectory_x,complete_skeleton_trajectory_y,complete_skeleton_trajectory_z,label=id_position_pairs[l][0])
                    ax = plt.axes(projection ="3d")
                    ax.scatter(complete_skeleton_trajectory_x,complete_skeleton_trajectory_y,complete_skeleton_trajectory_z,label=id_position_pairs[l][0])
                    
                    ##################################################################
                    #plt.legend()
                    #plt.show() 
                    #print("a") 
                    ##################################################################
                    
                else:
                    list_index_key = len(set_of_available_nodes)
                    start_node = node(id_position_pairs[l][1][0],
                                    id_position_pairs[l][1][1],
                                    allPositions[l][0][jump_in_arrays[l][0]:jump_in_arrays[l][1]],
                                    allPositions[l][1][jump_in_arrays[l][0]:jump_in_arrays[l][1]],
                                    allSkeletons[l][jump_in_arrays[l][0]:jump_in_arrays[l][1]],
                                    id_position_pairs[l][0] + " " + str(1),
                                    list_index_key)
                    set_of_available_nodes.append(start_node)
                    list_index_key = len(set_of_available_nodes)
                    #end_node   = node(id_position_pairs[l][1][0],id_position_pairs[l][1][1],allPositions[l][0][jump_in_arrays[l][0]:jump_in_arrays[l][1]],allPositions[l][1][jump_in_arrays[l][0]:jump_in_arrays[l][1]],id_position_pairs[l][0] + " " + str(k),list_index_key)
                    end_node   = node(1799,
                                    1799,
                                    allPositions[l][0][jump_in_arrays[l][0]:jump_in_arrays[l][1]],
                                    allPositions[l][1][jump_in_arrays[l][0]:jump_in_arrays[l][1]],
                                    allSkeletons[l][jump_in_arrays[l][0]:jump_in_arrays[l][1]],
                                    id_position_pairs[l][0] + " " + str(1),
                                    list_index_key)
                    set_of_available_nodes.append(end_node)
                    path = shortest_path(start_node,end_node)

                    print("Output:",id_position_pairs[l][0] + " " + str(k),path)
                    print("Jumps Start:",0,1, jump_in_arrays[l][0],jump_in_arrays[l][1],id_position_pairs[l][0] + " " + str(1))
                    print(" ")
                    
                    set_of_available_nodes.remove(start_node)
                    set_of_available_nodes.remove(end_node)

                    gap_filler_x = []
                    gap_filler_y = []
                    gap_filler_skeleton = []

                    ####################################################################
                    if path != None:
                        for x in range(len(path)):
                            gap_node = path[x]
                            coordinates = gap_node.return_coordinates()
                            gap_filler_x += coordinates[0]
                            gap_filler_y += coordinates[1]
                            gap_filler_skeleton += coordinates[2]
                    ###################################################################

                    #arrays concatenaten & in scatter graph plotten 
                    complete_trajectory_x = allPositions[l][0][jump_in_arrays[l][0]:jump_in_arrays[l][1]] + gap_filler_x 
                    complete_trajectory_y = allPositions[l][1][jump_in_arrays[l][0]:jump_in_arrays[l][1]] + gap_filler_y
                    complete_skeleton_trajectory = allSkeletons[l][jump_in_arrays[l][0]:jump_in_arrays[l][1]] + gap_filler_skeleton
                    complete_data.append(complete_skeleton_trajectory)

                    #plt.scatter(complete_trajectory_x,complete_trajectory_y,label=id_position_pairs[l][0]) 
                    #plt.scatter(complete_skeleton_trajectory[:][:][0],complete_skeleton_trajectory[:][:][1],complete_skeleton_trajectory[:][:][2],label=id_position_pairs[l][0]) 
                    complete_skeleton_trajectory_x = []
                    complete_skeleton_trajectory_y = []
                    complete_skeleton_trajectory_z = []
                    print("lengths:",len(complete_skeleton_trajectory),len(complete_skeleton_trajectory[0]),len(complete_skeleton_trajectory[0][0]))
                    print("lengths og:",len(allSkeletons),len(allSkeletons[0]),len(allSkeletons[0][0]))
                    for h in range(len(complete_skeleton_trajectory)):
                        for g in range(32):
                            if complete_skeleton_trajectory[h][g][0] != 0 and complete_skeleton_trajectory[h][g][1] != 0 and complete_skeleton_trajectory[h][g][2] != 0:
                                complete_skeleton_trajectory_x.append(complete_skeleton_trajectory[h][g][0])
                                complete_skeleton_trajectory_y.append(complete_skeleton_trajectory[h][g][1])
                                complete_skeleton_trajectory_z.append(complete_skeleton_trajectory[h][g][2])
                    ax = plt.axes(projection ="3d")
                    ax.scatter3D(complete_skeleton_trajectory_x,complete_skeleton_trajectory_y,complete_skeleton_trajectory_z,label=id_position_pairs[l][0])

                    ##################################################################
                    #plt.legend()
                    #plt.show()  
                    #print("a")
                    ##################################################################
        #plt.legend()
        #plt.show()  
        #sequence_number += 1 
        return complete_data

if __name__ == "__main__":
    #joints_coords_within_frame = np.zeros((32,3),dtype=np.float32) #np.empty((32,3))
    #skeleton_over_all_frames = []
    #sequence_number = 70
    #x_max = 7
    #x_min = -9
    #y_max = 0
    #y_min = -8

    trainingData = []
    totalNumberOfSequences = 20
    limitations = np.zeros((totalNumberOfSequences,4))
    
    for sequence in range(totalNumberOfSequences):
        limitations[sequence] = (float('inf'),float('-inf'),float('inf'),float('-inf'))

    for sequenceNumber in range(totalNumberOfSequences):
        totalNumSkeletons = calc_number_of_skeletons(sequenceNumber)
        skeleton_over_all_frames = np.zeros((totalNumSkeletons,32,3))
        completeDataSequence = calc_skeleton_sequences(sequenceNumber,limitations[sequenceNumber])
        print("[Status] Working on Sequence ", sequenceNumber)
        for personData in range(len(completeDataSequence)):
            trainingData.append(completeDataSequence[personData])
            #print("Sequence",sequenceNumber,":",len(completeDataSequence[personData]),len(completeDataSequence[personData][0]),len(completeDataSequence[personData][0][0]))
    print("[Status] Starting Data Saving")
    store_data(trainingData)
    print("[Status] Finished Data Saving")