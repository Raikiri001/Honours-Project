import numpy as np

class RuleBasedTrajectoryClassifier:

    def __init__(self, u_turn_class_id, overtake_class_id):

        #Class Ids
        self.u_turn_class_id = u_turn_class_id
        self.overtake_class_id = overtake_class_id
        self.normal_driving_class_id = 0

        # Smoothing to get reliable headings
        self.SMOOTHING_WINDOW = 5
        self.LANE_WIDTH_METRES = 3.5

        # U Turn
        self.MIN_U_TURN_HEADING_REVERSAL_DEGREES = 150
        self.MAX_DISTANCE_BETWEEN_START_END = 20

        # Overtake
        self.MAX_NEIGHBOUR_DISTANCE_METRES = 5.25
        self.MIN_LANE_CHANGE_LATERAL_OFFSET_METRES = 1.75

    #Classifies u turn before overtake. Normal if U turn or overtake was not predicted
    def classify_trajectory(self, trajectory):

        delta_x, delta_y, distance_to_neighbour, relative_speed_to_neighbour = trajectory[:, 0], trajectory[:, 1], trajectory[:, 4], trajectory[:, 5]

        if self.is_u_turn(delta_x, delta_y):
            return self.u_turn_class_id

        elif self.is_overtake(delta_x, delta_y, distance_to_neighbour, relative_speed_to_neighbour):
            return self.overtake_class_id

        else:
            return self.normal_driving_class_id

    #U TURN
    def is_u_turn(self, delta_x, delta_y):
        if self.performed_a_heading_reversal(delta_x, delta_y) and self.returned_near_starting_position(delta_x, delta_y):
            return True
        return False

    #Checks if the heading difference between the start and end heading is at least MIN_U_TURN_HEADING_REVERSAL_DEGREES to be counted as a heading reversal
    def performed_a_heading_reversal(self, delta_x, delta_y):
        start_heading, end_heading = self.get_start_and_end_headings(delta_x, delta_y)

        heading_difference = np.degrees(self.wrap_heading_radians(end_heading - start_heading))

        if abs(heading_difference) >= self.MIN_U_TURN_HEADING_REVERSAL_DEGREES:
            return True
        return False

    # Checked if vehicle trajectory starts and ends within MAX_DISTANCE_BETWEEN_START_END
    def returned_near_starting_position(self, delta_x, delta_y):
        coordinates = self.get_vehicle_coordinates(delta_x, delta_y)
        start_coordinate, end_coordinate = self.get_start_and_end_coordinates(coordinates)

        displacement = end_coordinate - start_coordinate

        start_to_end_distance = np.linalg.norm(displacement)

        if start_to_end_distance <= self.MAX_DISTANCE_BETWEEN_START_END:
            return True
        
        return False

    #OVERTAKE
    def is_overtake(self, delta_x, delta_y, distance_to_neighbour, relative_speed_to_neighbour):
        if self.was_next_to_a_slower_neighbouring_vehicle(distance_to_neighbour, relative_speed_to_neighbour) and self.performed_lane_change_out_and_lane_change_in(delta_x, delta_y):
            return True
        return False

    # Checks if the vehicle trajectory has a neighbour within the MAX_NEIGHBOUR_DISTANCE_METRES distance while going faster than that neighbour
    def was_next_to_a_slower_neighbouring_vehicle(self, distance_to_neighbour, relative_speed_to_neighbour):
        was_close = distance_to_neighbour <= self.MAX_NEIGHBOUR_DISTANCE_METRES
        was_faster = relative_speed_to_neighbour > 0

        if np.any(was_close & was_faster):
            return True
        return False

    # Checks if a vehicle's lateral offset from a line made between it's smoothed start and end points has varied by at least MIN_LANE_CHANGE_LATERAL_OFFSET_METRES
    def performed_lane_change_out_and_lane_change_in(self, delta_x, delta_y):
        coordinates = self.get_vehicle_coordinates(delta_x, delta_y)

        start_coordinate, end_coordinate = self.get_start_and_end_coordinates(coordinates)

        displacement = end_coordinate - start_coordinate

        start_to_end_distance = np.linalg.norm(displacement)

        if start_to_end_distance == 0:
            return False

        trajectory_direction = displacement / start_to_end_distance

        perpendicular_direction = np.array([-trajectory_direction[1], trajectory_direction[0]])

        lateral_offsets = (coordinates - start_coordinate) @ perpendicular_direction

        if lateral_offsets.max() - lateral_offsets.min() >= self.MIN_LANE_CHANGE_LATERAL_OFFSET_METRES:
            return True
        return False

    # Gets heading in radians
    @staticmethod
    def wrap_heading_radians(heading):
        return (heading + np.pi) % (2 * np.pi) - np.pi

    # Gets the vehicle coordinates based on sum of delta x and delta y
    def get_vehicle_coordinates(self, delta_x, delta_y):
        return np.stack([np.cumsum(delta_x), np.cumsum(delta_y)], axis=1)

    # Get smoothed start and end headings that are not noisy
    def get_start_and_end_headings(self, delta_x, delta_y):
        start_heading = np.arctan2(np.sum(delta_y[:self.SMOOTHING_WINDOW]), np.sum(delta_x[:self.SMOOTHING_WINDOW]))
        end_heading = np.arctan2(np.sum(delta_y[-self.SMOOTHING_WINDOW:]), np.sum(delta_x[-self.SMOOTHING_WINDOW:]))
        return start_heading, end_heading

    # Get smoothed start and end coordinate of trajectory
    def get_start_and_end_coordinates(self, coordinates):
        start_coordinate = coordinates[:self.SMOOTHING_WINDOW].mean(axis=0)
        end_coordinate = coordinates[-self.SMOOTHING_WINDOW:].mean(axis=0)
        return start_coordinate, end_coordinate
