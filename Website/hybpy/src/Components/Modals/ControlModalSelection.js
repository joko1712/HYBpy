import React, { useEffect, useState } from "react";
import {
    Button,
    Checkbox,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    FormControlLabel,
} from "@mui/material";

const ControlModalSelection = ({ open, headers, handleClose, onSave }) => {
    const [localSelectedHeaders, setLocalSelectedHeaders] = useState([]);

    useEffect(() => {
        setLocalSelectedHeaders([]);
    }, [open]);

    const handleHeaderChange = (header) => {
        setLocalSelectedHeaders((prevSelected) => {
            if (prevSelected.includes(header)) {
                return prevSelected.filter((h) => h !== header);
            } else {
                return [...prevSelected, header];
            }
        });
    };

    const handleSave = () => {
        console.log("Selected Headers: ", localSelectedHeaders);
        onSave(localSelectedHeaders);
        handleClose();
    };

    return (
        <Dialog open={open} onClose={handleClose}>
            <DialogTitle>Select Control Variables ID</DialogTitle>
            <DialogContent>
                {headers.map((header) => (
                    <FormControlLabel
                        key={header}
                        control={
                            <Checkbox
                                checked={localSelectedHeaders.includes(header)}
                                onChange={() => handleHeaderChange(header)}
                            />
                        }
                        label={header}
                    />
                ))}
            </DialogContent>
            <DialogActions>
                <Button onClick={handleSave} color='primary'>
                    Save
                </Button>
                <Button onClick={handleClose} color='primary'>
                    Cancel
                </Button>
            </DialogActions>
        </Dialog>
    );
};

export default ControlModalSelection;
