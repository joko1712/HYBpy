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

const speciesColors = [
    "rgb(255, 99, 132)",
    "rgb(54, 162, 235)",
    "rgb(75, 192, 192)",
    "rgb(255, 206, 86)",
    "rgb(153, 102, 255)",
    "rgb(255, 159, 64)",
    "rgb(201, 203, 207)",
    "rgb(255, 99, 71)",
    "rgb(124, 252, 0)",
    "rgb(0, 255, 255)",
    "rgb(128, 0, 128)",
    "rgb(255, 215, 0)",
];

export const LineChart = ({ data }) => {
    const chartData = useMemo(() => {
        const labels = data.map((item) => item.time);
        const datasets = Object.keys(data[0])
            .filter((key) => key !== "time" && !key.startsWith("sd"))
            .map((key, index) => {
                return {
                    label: key,
                    data: data.map((item) => item[key]),
                    borderColor: speciesColors[index % speciesColors.length],
                    backgroundColor: speciesColors[index % speciesColors.length],
                };
            });

        return {
            labels,
            datasets,
        };
    }, [data]);

    return <Line data={chartData} />;
};

export default LineChart;
