def getSplittedData(selected_features, train_size, val_size, test_size):

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    # load data
    file_path = 'RandomForest Datenset 1.csv'
    data = pd.read_csv(file_path, sep="\t").dropna()

    # Nach Sequenzen gruppieren
    grouped = data.groupby("Sequence")
    # Erstellen einer Liste, in der jede Sequenz eine Gruppe ist
    grouped_data = [(seq, group) for seq, group in grouped]

    # Splitten (z.B. 80% Training, 10% Validation, 10% Test)
    # Erster Split: Trainings- und Testdaten (z.B. 80% Training, 20% Test)
    train_cur, test_groups = train_test_split(grouped_data, test_size=test_size, random_state=42)

    # Zweiter Split: Trainingsdaten in Training und Validierung aufteilen (z.B. 8/9) für Training und 1/9 für Validierung)
    train_groups, val_groups = train_test_split(train_cur, test_size=val_size/(1-test_size), random_state=42)  #1/9 * 0.9 = 0.1 für val

    # Schritt 3: Die Gruppendaten wieder in DataFrames konvertieren
    train_df = pd.concat([group for _, group in train_groups])
    test_df = pd.concat([group for _, group in test_groups])
    val_df = pd.concat([group for _, group in val_groups])

    y_scaler = MinMaxScaler()
    X_scaler = MinMaxScaler()

    # scale train
    y_train_unscaled = train_df['CCS'].values.reshape(-1, 1) * 1e40
    y_train = y_scaler.fit_transform(y_train_unscaled)
    X_train_unscaled = train_df[selected_features]
    X_train = pd.DataFrame(X_scaler.fit_transform(X_train_unscaled), columns=X_train_unscaled.columns)

    # scale validation
    y_val_unscaled = val_df['CCS'].values.reshape(-1, 1) * 1e40
    y_val = y_scaler.transform(y_val_unscaled)
    X_val_unscaled = val_df[selected_features]
    X_val = pd.DataFrame(X_scaler.transform(X_val_unscaled), columns=X_val_unscaled.columns)

    # scale test
    y_test_unscaled = test_df['CCS'].values.reshape(-1, 1) * 1e40
    y_test = y_scaler.transform(y_test_unscaled)
    X_test_unscaled = test_df[selected_features]
    X_test = pd.DataFrame(X_scaler.transform(X_test_unscaled), columns=X_test_unscaled.columns)




    return y_train, y_val, y_test, X_train, X_val, X_test


