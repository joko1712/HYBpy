import React, { useState, useEffect } from "react";
import {
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Grid,
    MenuItem,
    TextField,
    Tooltip,
} from "@mui/material";
import InfoIcon from "@mui/icons-material/Info";
import IconButton from "@mui/material/IconButton";
const HmodModal = ({ open, handleClose, handleSave, initialValues, setHmodOptions }) => {
    const [hiddenNodes, setHiddenNodes] = useState(initialValues ? initialValues.hiddenNodes : "");
    const [layer, setLayer] = useState(initialValues ? initialValues.layer : 1);
    const [tau, setTau] = useState(initialValues ? initialValues.tau : 0.25);
    const [mode, setMode] = useState(initialValues ? initialValues.mode : 1);
    const [method, setMethod] = useState(initialValues ? initialValues.method : 2);
    const [jacobian, setJacobian] = useState(initialValues ? initialValues.jacobian : 1);
    const [hessian, setHessian] = useState(initialValues ? initialValues.hessian : 0);
    const [niter, setNiter] = useState(initialValues ? initialValues.niter : 400);
    const [nstep, setNstep] = useState(initialValues ? initialValues.nstep : 2);
    const [bootstrap, setBootstrap] = useState(initialValues ? initialValues.bootstrap : 0);
    const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

    const toggleAdvancedSettings = () => {
        setShowAdvancedSettings(!showAdvancedSettings);
    };

    const handleSaveChanges = () => {
        const updatedOptions = {
            hiddenNodes: hiddenNodes.split(" ").map(Number),
            layer,
            tau,
            mode,
            method,
            jacobian,
            hessian,
            niter,
            nstep,
            bootstrap,
        };
        setHmodOptions(updatedOptions);
        handleSave(updatedOptions);
    };

    useEffect(() => {
        if (initialValues) {
            setHiddenNodes(initialValues.hiddenNodes);
            setLayer(initialValues.layer);
            setTau(initialValues.tau);
            setMode(initialValues.mode);
            setMethod(initialValues.method);
            setJacobian(initialValues.jacobian);
            setHessian(initialValues.hessian);
            setNiter(initialValues.niter);
            setNstep(initialValues.nstep);
            setBootstrap(initialValues.bootstrap);
        }
    }, [initialValues]);

    return (
        <Dialog open={open} onClose={handleClose}>
            <DialogTitle>Edit HMOD Settings</DialogTitle>

            <DialogContent>
                <Grid container spacing={2} marginTop={2}>
                    <Grid item xs={12}>
                        <TextField
                            style={{ width: "90%" }}
                            label='Hidden Nodes'
                            value={hiddenNodes}
                            onChange={(e) => setHiddenNodes(e.target.value)}
                            helperText="Enter hidden nodes as space-separated integers (e.g., '5 5' for 2 layers with 5 nodes each)"
                        />
                        <Tooltip
                            title='This specifies the number of neurons in each hidden layer of the neural network.'
                            arrow>
                            <IconButton size='small'>
                                <InfoIcon />
                            </IconButton>
                        </Tooltip>
                    </Grid>

                    <Grid item xs={12}>
                        <TextField
                            label='Layer'
                            select
                            style={{ width: "90%" }}
                            value={layer}
                            onChange={(e) => setLayer(Number(e.target.value))}
                            helperText='Select activation function. Options: Tanh, ReLu, LSTM'>
                            <MenuItem value={1}>Tanh</MenuItem>
                            <MenuItem value={2}>ReLu</MenuItem>
                            <MenuItem value={3}>LSTM</MenuItem>
                        </TextField>
                        <Tooltip
                            title='Tanh: Effective in neural networks for its ability to center data around zero. ReLU: Effective in avoiding the vanishing gradient problem make it ideal for deep networks. LSTMLayer: Powerful for tasks involving sequences, such as time-series prediction by introducing a memory cell that can maintain its state over long periods.'
                            arrow>
                            <IconButton size='small'>
                                <InfoIcon />
                            </IconButton>
                        </Tooltip>
                    </Grid>
                    <Grid item xs={12}>
                        <TextField
                            style={{ width: "90%" }}
                            label='Method'
                            select
                            value={method}
                            onChange={(e) => setMethod(Number(e.target.value))}
                            helperText='Select training method. Options: Trust Region Reflective(trf), trust-constr, Simulated Annealing, ADAM'>
                            <MenuItem value={1}>TRF</MenuItem>

                            <MenuItem value={2}>Trust-Constr</MenuItem>

                            <MenuItem value={3}>Dual Annealing</MenuItem>

                            <MenuItem value={4}>ADAM</MenuItem>
                        </TextField>
                        <Tooltip
                            title='Trust Region Reflective: Suitable for large-scale nonlinear least-squares problems. Trust-Region Constrained or L-BFGS-B: Suitable for constrained optimization problems. Simulated Annealing: Probabilistic technique for approximating the global optimum. ADAM: Optimization algorithm that computes adaptive learning rates for each parameter.'
                            arrow>
                            <IconButton size='small'>
                                <InfoIcon />
                            </IconButton>
                        </Tooltip>
                    </Grid>
                    <Grid item xs={12}>
                        <TextField
                            style={{ width: "90%" }}
                            label='Niter'
                            type='number'
                            value={niter}
                            onChange={(e) => setNiter(Number(e.target.value))}
                            helperText='Number of integrations'
                        />
                        <Tooltip
                            title='Number of iterations: Defines how many times the learning algorithm will work through the entire training dataset.'
                            arrow>
                            <IconButton size='small'>
                                <InfoIcon />
                            </IconButton>
                        </Tooltip>
                    </Grid>
                    <Grid item xs={12}>
                        <TextField
                            style={{ width: "90%" }}
                            label='Nstep'
                            type='number'
                            value={nstep}
                            onChange={(e) => setNstep(Number(e.target.value))}
                            helperText='Number of re-initializations'
                        />
                        <Tooltip
                            title='Number of re-initializations: Specifies how many times the model will re-initialize the weights during training to find the optimal values.'
                            arrow>
                            <IconButton size='small'>
                                <InfoIcon />
                            </IconButton>
                        </Tooltip>
                    </Grid>
                    <Grid item xs={12}>
                        <Button
                            fullWidth
                            variant='contained'
                            onClick={toggleAdvancedSettings}
                            sx={{ marginBottom: 2 }}>
                            {showAdvancedSettings
                                ? "Hide Advanced Settings"
                                : "Show Advanced Settings"}
                        </Button>
                    </Grid>

                    {showAdvancedSettings && (
                        <>
                            <Grid item xs={12}>
                                <TextField
                                    style={{ width: "90%" }}
                                    label='Jacobian'
                                    select
                                    value={jacobian}
                                    onChange={(e) => setJacobian(Number(e.target.value))}
                                    helperText='Direct calculation of jac'>
                                    <MenuItem value={0}>Off</MenuItem>
                                    <MenuItem value={1}>On</MenuItem>
                                </TextField>
                                <Tooltip
                                    title='Direct calculation of Jacobian: Enables or disables the direct calculation of the Jacobian matrix.'
                                    arrow>
                                    <IconButton size='small'>
                                        <InfoIcon />
                                    </IconButton>
                                </Tooltip>
                            </Grid>

                            <Grid item xs={12}>
                                <TextField
                                    style={{ width: "90%" }}
                                    label='Hessian'
                                    select
                                    value={hessian}
                                    onChange={(e) => setHessian(Number(e.target.value))}
                                    helperText='Direct calculation of Hess'>
                                    <MenuItem value={0}>Off</MenuItem>
                                    <MenuItem value={1}>On</MenuItem>
                                </TextField>
                                <Tooltip
                                    title='Direct calculation of Hessian: Enables or disables the direct calculation of the Hessian matrix.'
                                    arrow>
                                    <IconButton size='small'>
                                        <InfoIcon />
                                    </IconButton>
                                </Tooltip>
                            </Grid>

                            <Grid item xs={12}>
                                <TextField
                                    style={{ width: "90%" }}
                                    label='TAU'
                                    type='number'
                                    value={tau}
                                    onChange={(e) => setTau(Number(e.target.value))}
                                    helperText='Enter a floating-point value for the TAU parameter'
                                />
                                <Tooltip
                                    title='A parameter that influences the convergence speed of the optimization process.'
                                    arrow>
                                    <IconButton size='small'>
                                        <InfoIcon />
                                    </IconButton>
                                </Tooltip>
                            </Grid>

                            <Grid item xs={12}>
                                <TextField
                                    style={{ width: "90%" }}
                                    label='Mode'
                                    select
                                    value={mode}
                                    onChange={(e) => setMode(Number(e.target.value))}
                                    helperText='Trainning Mode: Indirect, Direct, Semidirect '>
                                    <MenuItem value={1}>Indirect</MenuItem>
                                    <MenuItem value={2}>Direct</MenuItem>
                                    <MenuItem value={3}>Semidirect</MenuItem>
                                </TextField>
                                <Tooltip
                                    title='Indirect: Separates the calculation of residuals and Jacobians from forward simulation. Direct: Fully integrates the calculation using automatic differentiation (not yet implemented). Semidirect: Combines aspects of both indirect and direct modes for efficient computation.'
                                    arrow>
                                    <IconButton size='small'>
                                        <InfoIcon />
                                    </IconButton>
                                </Tooltip>
                            </Grid>

                            <Grid item xs={12}>
                                <TextField
                                    style={{ width: "90%" }}
                                    label='Bootstrap'
                                    select
                                    value={bootstrap}
                                    onChange={(e) => setBootstrap(Number(e.target.value))}
                                    helperText='Set Bootstrap On or Off'>
                                    <MenuItem value={0}>On</MenuItem>
                                    <MenuItem value={1}>Off</MenuItem>
                                </TextField>
                                <Tooltip
                                    title='Enables or disables the use of bootstrap sampling, which is a method for estimating the distribution of a statistic by resampling with replacement.'
                                    arrow>
                                    <IconButton size='small'>
                                        <InfoIcon />
                                    </IconButton>
                                </Tooltip>
                            </Grid>
                        </>
                    )}
                </Grid>
            </DialogContent>
            <DialogActions>
                <Button onClick={handleSaveChanges} color='primary'>
                    Save
                </Button>
                <Button onClick={handleClose} color='primary'>
                    Cancel
                </Button>
            </DialogActions>
        </Dialog>
    );
};

export default HmodModal;
