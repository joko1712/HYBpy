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
}) => {
    const [nx, setNx] = useState(1);
    const [ny, setNy] = useState(1);
    const [xOptions, setXOptions] = useState([{ val: "" }]);
    const [yOptions, setYOptions] = useState([{ id: "" }]);

    const handleNxChange = (event) => {
        const newNx = Number(event.target.value);
        setNx(newNx);
        setXOptions(Array.from({ length: newNx }, (_, i) => xOptions[i] || { val: "" }));
    };

    const handleNyChange = (event) => {
        const newNy = Number(event.target.value);
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
        handleSave({ nx, ny, xOptions, yOptions });
        handleClose();
    };

    useEffect(() => {
        if (!open) {
            setNx(1);
            setNy(1);
            setXOptions([{ val: "" }]);
            setYOptions([{ id: "" }]);
        }
    }, [open]);

    return (
        <Dialog open={open} onClose={handleClose}>
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
                        />
                    </Grid>
                    {xOptions.map((x, index) => (
                        <React.Fragment key={index}>
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    label={`Value ${index + 1}`}
                                    select
                                    value={x.val}
                                    onChange={(e) =>
                                        handleXOptionChange(index, "val", e.target.value)
                                    }
                                    helperText='Select from species or control'>
                                    {[...speciesOptions, ...controlOptions].map((option) => (
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
                        />
                    </Grid>
                    {yOptions.map((y, index) => (
                        <React.Fragment key={index}>
                            <Grid item xs={12}>
                                <TextField
                                    fullWidth
                                    label={`Y ID ${index + 1}`}
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
