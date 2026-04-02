document.addEventListener("DOMContentLoaded", function () {
    var table = document.getElementById("pb-results-table");
    if (!table) return;

    // Strip HTML tags and parse numeric value; treat dashes / empty as -Infinity
    function parseCell(html) {
        var text = html.replace(/<[^>]*>/g, "").trim();
        if (text === "\u2013" || text === "-" || text === "") return -Infinity;
        var num = parseFloat(text);
        return isNaN(num) ? -Infinity : num;
    }

    new DataTable("#pb-results-table", {
        paging: false,
        info: false,
        searching: true,
        ordering: true,
        orderMulti: false,
        autoWidth: false,
        scrollX: true,
        order: [[21, "desc"]],   // default sort by Avg descending

        columnDefs: [
            // Metric name: sortable + searchable
            { targets: 0, orderable: true, searchable: true },
            // Category: sortable + searchable, hidden by default
            { targets: 1, orderable: true, searchable: true, visible: false },
            // All data columns: numeric sort, not text-searchable
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
                        columns: ":gt(1)"   // skip Metric and Category
                    }
                ]
            }
        }
    });
});
