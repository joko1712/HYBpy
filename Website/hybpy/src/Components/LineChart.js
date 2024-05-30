import React, { useMemo } from "react";
import { Line } from "react-chartjs-2";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export const LineChart = ({ data }) => {
    const chartData = useMemo(() => {
        const labels = data.map((item) => item.time);
        const datasets = Object.keys(data[0])
            .filter((key) => key.startsWith("sd"))
            .map((key, index) => {
                return {
                    label: key,
                    data: data.map((item) => item[key]),
                    borderColor: index % 2 === 0 ? "rgb(255, 99, 132)" : "rgb(54, 162, 235)",
                    backgroundColor:
                        index % 2 === 0 ? "rgba(255, 99, 132, 0.5)" : "rgba(54, 162, 235, 0.5)",
                };
            });

        return {
            labels,
            datasets,
        };
    }, [data]);

    return <Line data={chartData} />;
};
