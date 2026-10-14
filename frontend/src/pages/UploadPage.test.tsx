import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { renderRoute } from '../test/utils';

describe('Upload page', () => {
  it('blocks submission and explains, in plain words, what is missing', async () => {
    renderRoute('/upload');
    const submit = await screen.findByTestId('submit-upload');
    fireEvent.click(submit);
    const summary = await screen.findByTestId('error-summary');
    expect(within(summary).getByText('Choose the site you photographed.')).toBeInTheDocument();
    expect(
      within(summary).getByText('Add the white-light photo for this site.'),
    ).toBeInTheDocument();
    expect(
      within(summary).getByText('Add the violet-light (AF) photo for this site.'),
    ).toBeInTheDocument();
    expect(
      within(summary).getByText('Tick consent. We cannot analyse photos without it.'),
    ).toBeInTheDocument();
    expect(screen.getByTestId('consent')).toHaveAttribute('aria-invalid', 'true');
  });

  it('requires consent even when the site and both photos are provided', async () => {
    const user = userEvent.setup();
    renderRoute('/upload');
    await user.click(await screen.findByRole('button', { name: /Right lateral tongue/ }));
    expect(screen.getByText(/Reference site suggested: Left lateral tongue/)).toBeInTheDocument();
    await user.upload(
      screen.getByTestId('file-wl'),
      new File(['x'], 'wl.jpg', { type: 'image/jpeg' }),
    );
    await user.upload(
      screen.getByTestId('file-af'),
      new File(['x'], 'af.jpg', { type: 'image/jpeg' }),
    );
    await user.click(screen.getByTestId('submit-upload'));
    const summary = await screen.findByTestId('error-summary');
    expect(within(summary).getAllByRole('listitem')).toHaveLength(1);
    expect(summary).toHaveTextContent('Tick consent');
    await user.click(screen.getByTestId('consent'));
    await waitFor(() => expect(screen.queryByTestId('error-summary')).not.toBeInTheDocument());
  });

  it('marks synthetic sample files and switches the data source to synthetic', async () => {
    const user = userEvent.setup();
    renderRoute('/upload');
    await user.upload(
      await screen.findByTestId('file-wl'),
      new File(['x'], 'SYNTHETIC_lateral_tongue_right_wl.jpg', { type: 'image/jpeg' }),
    );
    expect(screen.getByText(/File names mark these captures as SYNTHETIC/)).toBeInTheDocument();
    expect(screen.getByLabelText('Data source')).toHaveValue('synthetic');
  });
});
