    raw_dist = map_coordinates(df, np.array([[wy], [wx]]), order=1, mode='nearest')[0]
    