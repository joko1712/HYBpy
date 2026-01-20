import React from "react";
import {
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Grid,
    Typography,
} from "@mui/material";

const CLOSE_LOCK_MS = 10_000;

const TrainingModal = ({ open, handleClose }) => {
    const [closeLocked, setCloseLocked] = React.useState(false);
    const [secondsLeft, setSecondsLeft] = React.useState(0);

    React.useEffect(() => {
        if (!open) return;
        setCloseLocked(true);
        setSecondsLeft(Math.ceil(CLOSE_LOCK_MS / 1000));

        const start = Date.now();

        const intervalId = window.setInterval(() => {
            const elapsed = Date.now() - start;
            const remainingMs = Math.max(0, CLOSE_LOCK_MS - elapsed);
            setSecondsLeft(Math.ceil(remainingMs / 1000));

            if (remainingMs <= 0) {
                setCloseLocked(false);
                window.clearInterval(intervalId);
            }
        }, 250);

        return () => {
            window.clearInterval(intervalId);
        };
    }, [open]);

    const onDialogClose = (_, reason) => {
        if (closeLocked && (reason === "backdropClick" || reason === "escapeKeyDown")) {
            return;
        }
        handleClose();
    };

    return (
        <Dialog
            open={open}
            onClose={onDialogClose}
            disableEscapeKeyDown={closeLocked}
        >
            <DialogTitle>Training in Progress</DialogTitle>

            <DialogContent>
                <Grid container spacing={2} marginTop={2}>
                    <Grid item xs={12}>
                        <Typography sx={{ mt: 2 }}>
                            Your training has started. You will be redirected to the dashboard once you
                            close this modal.
                        </Typography>

                        {closeLocked && (
                            <Typography sx={{ mt: 2 }} color='text.secondary'>
                                Please wait {secondsLeft}s before closing.
                            </Typography>
                        )}
                    </Grid>
                </Grid>
            </DialogContent>

            <DialogActions>
                <Button
                    onClick={handleClose}
                    color='primary'
                    disabled={closeLocked}
                >
                    {closeLocked ? `Close (${secondsLeft}s)` : "Close"}
                </Button>
            </DialogActions>
        </Dialog>
    );
};

export default TrainingModal;
