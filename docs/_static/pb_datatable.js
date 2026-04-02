document.addEventListener("DOMContentLoaded", function () {
    var table = document.getElementById("pb-results-table");
    if (!table) return;

    // Column indices (0-based)
    // 0: Metric, 1: Category, 2–20: data columns, 21: Avg
    var MC_COLS   = [2, 4, 6, 8, 10, 13, 16, 18];          // columns ending in "MC"
    var EX_COLS   = [3, 5, 7, 9, 11, 14, 17, 19];           // columns ending in "EX"
    var FULL_COLS = [12, 15, 20];                             // columns ending in "Full"
    var ALL_DATA  = MC_COLS.concat(EX_COLS, FULL_COLS);       // all 19 data columns
    var AVG_COL   = 21;

    // Strip HTML tags and parse numeric value; treat dashes / empty as -Infinity
    function parseCell(html) {
        var text = html.replace(/<[^>]*>/g, "").trim();
        if (text === "\u2013" || text === "-" || text === "") return -Infinity;
        var num = parseFloat(text);
        return isNaN(num) ? -Infinity : num;
    }

    // Show only the given data columns (plus Metric, Category, Avg always visible)
    function showOnly(dt, cols) {
        ALL_DATA.concat([AVG_COL]).forEach(function (i) {
            dt.column(i).visible(cols.indexOf(i) !== -1 || i === AVG_COL);
        });
    }

    var dt = new DataTable("#pb-results-table", {
        paging: false,
        info: false,
        searching: true,
        ordering: true,
        orderMulti: false,
        autoWidth: false,
        scrollX: true,
        order: [[AVG_COL, "desc"]],

        columnDefs: [
            { targets: 0, orderable: true, searchable: true },
            { targets: 1, orderable: true, searchable: true, visible: false },
            {
                targets: "_all",
                orderable: true,
                searchable: false,
                type: "num",
                render: function (data, type) {
                    if (type === "sort" || type === "type") {
                        return parseCell(data);
                    }
                    return data;
                }
            }
        ],

        layout: {
            topStart: {
                buttons: [
                    {
                        extend: "colvis",
                        text: "Show / Hide Columns",
                        columns: ":gt(1)"
                    },
                    {
                        text: "All Columns",
                        action: function () { showOnly(dt, ALL_DATA); }
                    },
                    {
                        text: "MC Only",
                        action: function () { showOnly(dt, MC_COLS); }
                    },
                    {
                        text: "EX / Full Only",
                        action: function () { showOnly(dt, EX_COLS.concat(FULL_COLS)); }
                    }
                ]
            }
        }
    });
});
