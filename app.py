# --- Tab 1: Huấn luyện ---
with tab1:
    st.subheader("📡 Huấn luyện mô hình với dữ liệu mô phỏng hoặc thực tế")

    option = st.radio("Chọn nguồn dữ liệu huấn luyện:", ("Sinh dữ liệu mô phỏng", "Tải file Excel dữ liệu thực tế"))

    df = None  # Đặt mặc định tránh lỗi NameError

    if option == "Sinh dữ liệu mô phỏng":
        if st.button("Huấn luyện mô hình từ dữ liệu mô phỏng"):
            st.info("Đang sinh dữ liệu mô phỏng...")
            np.random.seed(42)
            n_samples = 1000  # Tạo 1000 mẫu dữ liệu mô phỏng
            data = []
            for _ in range(n_samples):
                lat_tx = np.random.uniform(10.0, 21.0)
                lon_tx = np.random.uniform(105.0, 109.0)
                lat_rx = lat_tx + np.random.uniform(-0.05, 0.05)
                lon_rx = lon_tx + np.random.uniform(-0.05, 0.05)
                h_rx = np.random.uniform(2, 100)
                freq = np.random.uniform(30, 2600)

                azimuth = calculate_azimuth(lat_rx, lon_rx, lat_tx, lon_tx)
                distance = sqrt((lat_tx - lat_rx)**2 + (lon_tx - lon_rx)**2) * 111
                signal = simulate_signal_strength(distance, h_rx, freq)

                data.append({
                    "lat_receiver": lat_rx,
                    "lon_receiver": lon_rx,
                    "antenna_height": h_rx,
                    "azimuth": azimuth,
                    "frequency": freq,
                    "signal_strength": signal,  # Đơn vị dBµV/m
                    "distance_km": distance
                })

            df = pd.DataFrame(data)
            st.success("Dữ liệu mô phỏng đã được sinh thành công!")

            # Hiển thị 5 dòng đầu tiên của dữ liệu mô phỏng
            st.dataframe(df.head())

            # Tạo file Excel để tải xuống
            towrite = BytesIO()
            df.to_excel(towrite, index=False, engine='openpyxl')
            towrite.seek(0)
            st.download_button(
                label="📥 Tải dữ liệu mô phỏng (.xlsx)",
                data=towrite,
                file_name="simulation_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        uploaded_data = st.file_uploader("📂 Tải file Excel dữ liệu thực tế", type=["xlsx"])
        if uploaded_data:
            df = pd.read_excel(uploaded_data)
            st.success("Đã tải dữ liệu thực tế.")
            st.dataframe(df.head())  # Hiển thị dữ liệu thực tế tải lên
        else:
            st.info("Vui lòng tải file dữ liệu để huấn luyện.")

    if df is not None and st.button("🔧 Tiến hành huấn luyện mô hình"):
        try:
            st.info("Đang huấn luyện mô hình...")

            # Xử lý thêm dữ liệu
            df['azimuth_sin'] = np.sin(np.radians(df['azimuth']))
            df['azimuth_cos'] = np.cos(np.radians(df['azimuth']))

            X = df[['lat_receiver', 'lon_receiver', 'antenna_height', 'signal_strength', 'frequency', 'azimuth_sin', 'azimuth_cos']]
            y = df[['distance_km']]

            # Chia dữ liệu thành tập huấn luyện và kiểm tra
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Tạo mô hình Neural Network đơn giản
            model = Sequential()
            model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # Lớp đầu vào với 64 đơn vị
            model.add(Dense(32, activation='relu'))  # Lớp ẩn với 32 đơn vị
            model.add(Dense(1))  # Lớp đầu ra với 1 đơn vị (kết quả khoảng cách dự đoán)

            # Biên dịch mô hình
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Huấn luyện mô hình
            model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

            # Đánh giá mô hình
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.success(f"Huấn luyện xong - MAE khoảng cách: {mae:.3f} km")
            st.success(f"RMSE: {rmse:.3f} km")
            st.success(f"R²: {r2:.3f}")

            buffer = BytesIO()
            joblib.dump(model, buffer)  # Lưu mô hình mạng nơ-ron
            buffer.seek(0)

            # Cung cấp nút tải mô hình đã huấn luyện
            st.download_button(
                label="📥 Tải mô hình huấn luyện (.joblib)",
                data=buffer,
                file_name="distance_model.joblib",
                mime="application/octet-stream"
            )
        except Exception as e:
            st.error(f"Đã xảy ra lỗi trong quá trình huấn luyện: {e}")
            st.exception(e)
with tab2:
    st.subheader("📍 Dự đoán tọa độ nguồn phát xạ")

    uploaded_model = st.file_uploader("📂 Tải mô hình đã huấn luyện (.joblib)", type=["joblib"])
    if uploaded_model:
        model = joblib.load(uploaded_model)

        uploaded_excel = st.file_uploader("📄 Hoặc tải file Excel chứa thông tin các trạm thu", type=["xlsx"])

        if uploaded_excel:
            df_input = pd.read_excel(uploaded_excel)
            results = []
            m = folium.Map(location=[df_input['lat_receiver'].mean(), df_input['lon_receiver'].mean()], zoom_start=8)

            for _, row in df_input.iterrows():
                az_sin = np.sin(np.radians(row['azimuth']))
                az_cos = np.cos(np.radians(row['azimuth']))
                X_input = np.array([[row['lat_receiver'], row['lon_receiver'], row['antenna_height'], row['signal_strength'], row['frequency'], az_sin, az_cos]])
                predicted_distance = model.predict(X_input)[0]
                predicted_distance = max(predicted_distance, 0.1)

                lat_pred, lon_pred = calculate_destination(row['lat_receiver'], row['lon_receiver'], row['azimuth'], predicted_distance)

                # Thêm thông tin về tần số và mức tín hiệu vào tooltip của "Nguồn phát dự đoán"
                folium.Marker(
                    [lat_pred, lon_pred],
                    tooltip=f"Nguồn phát dự đoán\nTần số: {row['frequency']} MHz\nMức tín hiệu: {row['signal_strength']} dBµV/m",
                    icon=folium.Icon(color='red')
                ).add_to(m)

                folium.Marker([row['lat_receiver'], row['lon_receiver']], tooltip="Trạm thu", icon=folium.Icon(color='blue')).add_to(m)
                folium.PolyLine(locations=[[row['lat_receiver'], row['lon_receiver']], [lat_pred, lon_pred]], color='green').add_to(m)

                results.append({
                    "lat_receiver": row['lat_receiver'],
                    "lon_receiver": row['lon_receiver'],
                    "lat_pred": lat_pred,
                    "lon_pred": lon_pred,
                    "predicted_distance_km": predicted_distance,
                    "frequency": row['frequency'],
                    "signal_strength": row['signal_strength']
                })

            st.dataframe(pd.DataFrame(results))
            st_folium(m, width=800, height=500)
