function results = test_fade(inputFolder)
%TEST_FADE Evaluate FADE on the immediate images in a folder.
%   results = test_fade('path/to/images');
%   With no argument, evaluate the included 31 CoA++ examples.
%   Returns a table and prints the mean. No images are resized or modified.

matlabFolder = fileparts(mfilename('fullpath'));
if nargin < 1
    inputFolder = fullfile(fileparts(matlabFolder), 'examples', 'Fattal', 'CoA++');
end
[ok, folderInfo] = fileattrib(inputFolder);
assert(ok && folderInfo.directory, 'Input must be an existing image folder.');
inputFolder = folderInfo.Name;
files = dir(inputFolder);
extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'};
keep = false(size(files));
for k = 1:numel(files)
    [~, ~, extension] = fileparts(files(k).name);
    keep(k) = ~files(k).isdir && any(strcmpi(extension, extensions));
end
files = files(keep);
[~, order] = sort({files.name});
files = files(order);
assert(~isempty(files), 'No supported images in the folder.');

previousFolder = pwd;
previousPath = path;
previousWarnings = warning;
workFolder = tempname;
mkdir(workFolder);
cleanup = onCleanup(@()restore_state(previousFolder, previousPath, previousWarnings, workFolder)); %#ok<NASGU>
copyfile(fullfile(matlabFolder, 'FADE', '*'), workFolder);
addpath(workFolder, '-begin');
cd(workFolder);

names = strings(numel(files), 1);
values = nan(numel(files), 1);
errors = strings(numel(files), 1);
for k = 1:numel(files)
    names(k) = string(files(k).name);
    try
        [I, map] = imread(fullfile(inputFolder, files(k).name));
        if ~isempty(map), I = im2uint8(ind2rgb(I, map)); end
        assert(isa(I, 'uint8'), 'Convert non-uint8 images explicitly before evaluation.');
        if ismatrix(I), I = repmat(I, 1, 1, 3); end
        assert(size(I, 3) == 3, 'Expected an RGB image.');
        value = FADE(I);
        assert(isnumeric(value) && isscalar(value) && isreal(value) && isfinite(value), 'Invalid score.');
        values(k) = double(value);
    catch exception
        errors(k) = string(exception.message);
    end
end
results = table(names, values, errors, 'VariableNames', {'Image', 'FADE', 'Error'});
valid = isfinite(values);
fprintf('FADE mean = %.8f (%d/%d images)\n', mean(values(valid)), nnz(valid), numel(values));
end

function restore_state(previousFolder, previousPath, previousWarnings, workFolder)
cd(previousFolder);
path(previousPath);
warning(previousWarnings);
if isfolder(workFolder)
    rmdir(workFolder, 's');
end
end
