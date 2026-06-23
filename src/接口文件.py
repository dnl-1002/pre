from nbeats.nbeats_model import predict_next_points

# 使用示例
if __name__ == "__main__":
    # 示例数据
    bio_i_str = 'Copepoda'
    example_data = {
        'y': [1.1333, 1.4667, 1.0, 0.9333, 1.8667, 1.5333, 1.2667, 0.9333, 2.4667, 1.9333, 2.7333, 1.6],  # 桡足类数量
        'Temperature': [19.64, 19.7525, 19.835, 19.8825, 19.9275, 19.915, 20.0525, 20.315, 20.405, 20.445, 20.13, 19.995],  # 温度
        'Salinity': [33.095, 33.1275, 33.1725, 33.1675, 33.125, 33.0625, 33.0625, 33.13, 33.13, 33.03, 32.965, 33.01]  # 盐度
    }

    try:
        # 获取预测值
        predictions = predict_next_points(example_data,bio_i_str,model_dir='saved_models')
        print("\n预测的未来1个时间点的桡足类数量:")
        print(f"预测值: {predictions[-1]:.2f}")

        # 可视化结果
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        # 绘制历史数据
        plt.plot(range(12), example_data['y'], 'b-', label='history')

        # 获取历史数据的最后一个值
        last_historical_value = example_data['y'][-1]

        # 创建连接点的x和y值
        x_connect = [11, 12]
        y_connect = [last_historical_value, predictions[-1]]

        # 绘制预测线（包含连接部分）
        plt.plot(x_connect, y_connect, 'r--', label='prediction')

        plt.xlabel('hours')
        plt.ylabel('copepoda')
        plt.title('copepoda_prediction')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"运行示例时出现错误: {str(e)}")