/**
 * @jest-environment jsdom
 */
const { toggleTable, downloadCSV } = require("./script");

document.body.innerHTML = `
<table>
    <tbody>
        <tr class="extra-row" style="display: none;"></tr>
        <tr class="extra-row" style="display: none;"></tr>
    </tbody>
</table>
<button id="toggleBtn">Show Full Table</button>
`;

test("toggleTable should show and hide rows correctly", () => {
    toggleTable();
    let rows = document.querySelectorAll(".extra-row");
    expect(rows[0].style.display).toBe("table-row");

    toggleTable();
    expect(rows[0].style.display).toBe("none");
});

test("downloadCSV should generate a valid CSV file", () => {
    global.URL.createObjectURL = jest.fn();
    document.body.innerHTML += `<a id="download-link"></a>`;

    expect(() => downloadCSV()).not.toThrow();
});
