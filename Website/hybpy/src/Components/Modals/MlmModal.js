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

const MlmModal = ({
    open,
    handleClose,
    handleSave,
    speciesOptions,
    controlOptions,
    parameterOptions,
    compartmentOptions,
}) => {
    const [nx, setNx] = useState(1);
    const [ny, setNy] = useState(1);
    const [xOptions, setXOptions] = useState([{ val: "" }]);
    const [yOptions, setYOptions] = useState([{ id: "" }]);
    const [error, setError] = useState(false);

    const handleNxChange = (event) => {
        const newNx = Math.max(0, Number(event.target.value));
        setNx(newNx);
        setXOptions(Array.from({ length: newNx }, (_, i) => xOptions[i] || { val: "" }));
    };

    const handleNyChange = (event) => {
        const newNy = Math.max(0, Number(event.target.value));
        setNy(newNy);
        setYOptions(Array.from({ length: newNy }, (_, i) => yOptions[i] || { id: "" }));
    };

    const handleXOptionChange = (index, key, value) => {
        const newXOptions = [...xOptions];
        newXOptions[index][key] = value;
        setXOptions(newXOptions);
    };

    const handleYOptionChange = (index, key, value) => {
        const newYOptions = [...yOptions];
        newYOptions[index][key] = value;
        setYOptions(newYOptions);
    };

    const handleSaveChanges = () => {
        const allInputsSelected = xOptions.every((x) => x.val !== "");
        const allOutputsSelected = yOptions.every((y) => y.id !== "");

        if (nx > 0 && !allInputsSelected) {
            setError(true);
            return;
        }
        if (ny > 0 && !allOutputsSelected) {
            setError(true);
            return;
        }

        setError(false);
        handleSave({ nx, ny, xOptions, yOptions });
        handleClose();
    };

    useEffect(() => {
        if (!open) {
            setNx(1);
            setNy(1);
            setXOptions([{ val: "" }]);
            setYOptions([{ id: "" }]);
            setError(false);
        }
    }, [open]);

    return (
        <Dialog
            open={open}
            disableEscapeKeyDown
            PaperProps={{
                onClick: (e) => e.stopPropagation(),
            }}>
            <DialogTitle>Add ML Block to HMOD</DialogTitle>
            <DialogContent>
                <Grid container spacing={2} marginTop={2}>
                    <Grid item xs={12}>
                        <TextField
                            fullWidth
                            label='Number of Inputs'
                            type='number'
                            value={nx}
                            onChange={handleNxChange}
                            error={error && nx > 0 && !xOptions.every((x) => x.val !== "")}
                            helperText={
                                error && nx > 0 && !xOptions.every((x) => x.val !== "")
                                    ? "Please select all input options"
                                    : ""
                            }
                        />
                    </Grid>
                    {xOptions.map((x, index) => (
                        <React.Fragment key={index}>
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    label={`Input ID ${index + 1}`}
                                    select
                                    value={x.val}
                                    onChange={(e) =>
                                        handleXOptionChange(index, "val", e.target.value)
                                    }
                                    helperText='Select from species, control or compartments'>
                                    {[
                                        ...speciesOptions,
                                        ...controlOptions,
                                        ...compartmentOptions,
                                    ].map((option) => (
                                        <MenuItem key={option} value={option}>
                                            {option}
                                        </MenuItem>
                                    ))}
                                </TextField>
                            </Grid>
                        </React.Fragment>
                    ))}
                    <Grid item xs={12}>
                        <TextField
                            fullWidth
                            label='Number of Outputs'
                            type='number'
                            value={ny}
                            onChange={handleNyChange}
                            error={error && ny > 0 && !yOptions.every((y) => y.id !== "")}
                            helperText={
                                error && ny > 0 && !yOptions.every((y) => y.id !== "")
                                    ? "Please select all output options"
                                    : ""
                            }
                        />
                    </Grid>
                    {yOptions.map((y, index) => (
                        <React.Fragment key={index}>
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    label={`Output ID ${index + 1}`}
                                    select
                                    value={y.id}
                                    onChange={(e) =>
                                        handleYOptionChange(index, "id", e.target.value)
                                    }
                                    helperText='Select from parameters'>
                                    {parameterOptions.map((option) => (
                                        <MenuItem key={option} value={option}>
                                            {option}
                                        </MenuItem>
                                    ))}
                                </TextField>
                            </Grid>
                        </React.Fragment>
                    ))}
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

export default MlmModal;
