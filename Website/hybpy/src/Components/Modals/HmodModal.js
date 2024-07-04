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
} from "@mui/material";

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
            <DialogTitle>Update HMOD Options</DialogTitle>
            <DialogContent>
                <Grid container spacing={2} marginTop={2}>
                    <Grid item xs={12}>
                        <TextField
                            fullWidth
                            label='Hidden Nodes'
                            value={hiddenNodes}
                            onChange={(e) => setHiddenNodes(e.target.value)}
                            helperText="Enter hidden nodes as space-separated integers (e.g., '5 5' for 2 layers with 5 nodes each)"
                        />
                    </Grid>
                    <Grid item xs={12}>
                        <TextField
                            fullWidth
                            label='Layer'
                            select
                            value={layer}
                            onChange={(e) => setLayer(Number(e.target.value))}
                            helperText='Activation function'>
                            <MenuItem value={1}>Tanh</MenuItem>
                            <MenuItem value={2}>ReLu</MenuItem>
                        </TextField>
                    </Grid>
                    <Grid item xs={12}>
                        <TextField
                            fullWidth
                            label='Method'
                            select
                            value={method}
                            onChange={(e) => setMethod(Number(e.target.value))}
                            helperText='Trainning Method: Trust Region Reflective(trf), trust-constr, ADAM'>
                            <MenuItem value={1}>trf</MenuItem>
                            <MenuItem value={2}>trust-constr</MenuItem>
                            <MenuItem value={3}>ADAM</MenuItem>
                        </TextField>
                    </Grid>
                    <Grid item xs={12}>
                        <TextField
                            fullWidth
                            label='Niter'
                            type='number'
                            value={niter}
                            onChange={(e) => setNiter(Number(e.target.value))}
                            helperText='Number of integrations'
                        />
                    </Grid>
                    <Grid item xs={12}>
                        <TextField
                            fullWidth
                            label='Nstep'
                            type='number'
                            value={nstep}
                            onChange={(e) => setNstep(Number(e.target.value))}
                            helperText='Number of re-initializations'
                        />
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
                                    fullWidth
                                    label='Jacobian'
                                    select
                                    value={jacobian}
                                    onChange={(e) => setJacobian(Number(e.target.value))}
                                    helperText='Direct calculation of jac'>
                                    <MenuItem value={0}>Off</MenuItem>
                                    <MenuItem value={1}>On</MenuItem>
                                </TextField>
                            </Grid>
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    label='Hessian'
                                    select
                                    value={hessian}
                                    onChange={(e) => setHessian(Number(e.target.value))}
                                    helperText='Direct calculation of Hess'>
                                    <MenuItem value={0}>Off</MenuItem>
                                    <MenuItem value={1}>On</MenuItem>
                                </TextField>
                            </Grid>
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    label='TAU'
                                    type='number'
                                    value={tau}
                                    onChange={(e) => setTau(Number(e.target.value))}
                                    helperText='Enter a floating-point value'
                                />
                            </Grid>
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    label='Mode'
                                    select
                                    value={mode}
                                    onChange={(e) => setMode(Number(e.target.value))}
                                    helperText='Trainning Mode: Indirect, Direct, Semidirect'>
                                    <MenuItem value={1}>Indirect</MenuItem>
                                    <MenuItem value={2}>Direct</MenuItem>
                                    <MenuItem value={3}>Semidirect</MenuItem>
                                </TextField>
                            </Grid>
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    label='Bootstrap'
                                    select
                                    value={bootstrap}
                                    onChange={(e) => setBootstrap(Number(e.target.value))}
                                    helperText='Set Bootstrap On or Off'>
                                    <MenuItem value={0}>On</MenuItem>
                                    <MenuItem value={1}>Off</MenuItem>
                                </TextField>
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
