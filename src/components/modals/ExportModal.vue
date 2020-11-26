<template>
  <div>
    Export the selected elements
    <CheckboxList label="Models" :children="modelEditors.map((child) => child.name)"
                  v-show="modelEditors.length > 0"/>
    <CheckboxList label="Datasets" :children="dataEditors.map((child) => child.name)"
                  v-show="dataEditors.length > 0"/>
    <CheckboxList label="Codevault" :children="files.map((child) => child.filename)"
                  v-show="files.length > 0"/>
    <div class="buttons">
      <UIButton text="Cancel" @click="cancel"/>
      <UIButton text="Save" :primary="true"/>
    </div>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';
import Modal from '@/components/modals/Modal.vue';
import UIButton from '@/components/buttons/UIButton.vue';
import CheckboxList from '@/components/CheckboxList.vue';
import { mapGetters } from 'vuex';
import CheckboxValue from '@/baklava/CheckboxValue';

@Component({
  components: { CheckboxList, UIButton, Modal },
  computed: mapGetters(['modelEditors', 'dataEditors', 'files']),
})
export default class ExportModal extends Vue {
  private open = true;
  private modelsChecked = CheckboxValue.CHECKED;

  private cancel() {
    this.$parent.$emit('input', false);
  }
}
</script>

<style scoped>
.buttons {
  display: flex;
  float: right;
  margin-top: 0.75rem;
}

.button {
  margin-left: 1rem;
}
</style>
